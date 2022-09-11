import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
from torch.nn import ModuleList
from torch.nn.functional import one_hot
from transformers import BertModel, BertConfig


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class Chemical_Multimodal_Model(torch.nn.Module):
    def __init__(self, graph_encoder_params, smiles_encoder_params):
        super(Chemical_Multimodal_Model, self).__init__()
        # graph
        self.graph_encoder_n_layer = graph_encoder_params['n_layer']
        self.graph_encoder_dim = graph_encoder_params['hidden_dim']
        self.num_atom_type = graph_encoder_params['num_atom_type']
        self.gnn_layers = ModuleList([])
        for i in range(self.graph_encoder_n_layer):
                self.gnn_layers.append(GraphConv(in_feats=self.graph_encoder_dim,
                                                 out_feats=self.graph_encoder_dim,
                                                 norm='both'))
        self.pooling_layer = AvgPooling()
        self.embedding_h = nn.Embedding(self.num_atom_type, self.graph_encoder_dim)
        # self.embedding_e = nn.Linear(4, 64)
        # text
        self.smiles_encoder_n_layer = smiles_encoder_params['n_layer']
        self.smiles_encoder_hidden_dim = smiles_encoder_params['hidden_dim']
        bert_config = BertConfig(vocab_size=3132,
                                 hidden_size=self.smiles_encoder_hidden_dim,
                                 num_hidden_layers=self.smiles_encoder_n_layer,
                                 num_attention_heads=self.smiles_encoder_hidden_dim // 8,
                                 intermediate_size=self.smiles_encoder_hidden_dim * 4)

        self.smiles_encoder = BertModel(bert_config)
        self.text_projection = torch.nn.Linear(self.smiles_encoder_hidden_dim, self.smiles_encoder_hidden_dim)
        self.MLP_layer = MLPReadout(self.smiles_encoder_hidden_dim * 2, 1)

    def forward(self, graph, smiles_inputs):
        h = graph.ndata['f']
        h = self.embedding_h(h) # [2344] -> [2344, 64]
        # e = graph.edata['f']
        # e = self.embedding_e(e)
        for layer in self.gnn_layers:
            h = layer(graph, h) # [2344, 64] -> [2344, 64]
        graph_embedding = self.pooling_layer(graph, h) # [2344, 64] -> [128, 64]
        text_embedding = self.smiles_encoder(smiles_inputs['input_ids'],
                                      smiles_inputs['attention_mask'],
                                      smiles_inputs['token_type_ids'])['pooler_output'] # [128, 64]
        text_embedding = self.text_projection(text_embedding)

        embeddings = torch.cat([graph_embedding, text_embedding], axis=1)
        outputs = self.MLP_layer(embeddings)

        return outputs
