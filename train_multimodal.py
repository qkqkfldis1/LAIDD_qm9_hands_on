import os
import torch
import numpy as np
import argparse
from torch import nn, optim

from qm9_dataset import QM9DGLDataset, collate_multimodal
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from multi_modal_modeling import Chemical_Multimodal_Model
import torch.nn as nn
from tqdm import tqdm
from tokenizer import SMILES_SPE_Tokenizer
tokenizer = SMILES_SPE_Tokenizer(vocab_file='vocab_spe.txt', spe_file= 'SPE_ChEMBL.txt')


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def to_np(x):
    return x.cpu().detach().numpy()


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS, device):
    model.train()

    num_iters = len(dataloader)
    for i, (g, y, smiles) in enumerate(tqdm(dataloader, total=len(dataloader))):
        g = g.to(device)
        y = y.to(device)
        smiles = tokenizer(smiles, padding=True, max_length=12, truncation=True, return_tensors='pt')
        smiles_input = {'input_ids': smiles['input_ids'].to(device),
                        'attention_mask': smiles['attention_mask'].to(device),
                        'token_type_ids': smiles['token_type_ids'].to(device)}

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(g, smiles_input)
        l1_loss, __, rescale_loss = loss_fnc(pred, y)

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.train_params.print_epoch_interval == 0:
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]")

        scheduler.step(epoch + i / num_iters)


def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS, device):
    model.eval()

    rloss = 0
    for i, (g, y, smiles) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)
        smiles = tokenizer(smiles, padding=True, max_length=12, truncation=True, return_tensors='pt')
        smiles_input = {'input_ids': smiles['input_ids'].to(device),
                        'attention_mask': smiles['attention_mask'].to(device),
                        'token_type_ids': smiles['token_type_ids'].to(device)}
        # run model forward and compute loss
        pred = model(g, smiles_input).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.val_size

    print(f"...[{epoch}|val] rescale loss: {rloss:.5f} [units]")


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, device):
    model.eval()

    rloss = 0
    for i, (g, y, smiles) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)
        smiles = tokenizer(smiles, padding=True, max_length=12, truncation=True, return_tensors='pt')
        smiles_input = {'input_ids': smiles['input_ids'].to(device),
                        'attention_mask': smiles['attention_mask'].to(device),
                        'token_type_ids': smiles['token_type_ids'].to(device)}

        # run model forward and compute loss
        pred = model(g, smiles_input).detach()
        __, __, rl = loss_fnc(pred, y, use_mean=False)
        rloss += rl
    rloss /= FLAGS.test_size

    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")



################ 1. parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/multi_modal.yaml', help='configration for model')
parser.add_argument('--pretrained_path', type=str, default=None, help='configration for model')
args, extra_args = parser.parse_known_args()
FLAGS = OmegaConf.load(args.config)

# Create model directory
if not os.path.isdir(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)


torch.manual_seed(FLAGS.train_params.seed)
np.random.seed(FLAGS.train_params.seed)

# Automatically choose GPU if available
device = gpu_setup(FLAGS['gpu']['use'], FLAGS['gpu']['id'])

################ 2. Prepare data
train_dataset = QM9DGLDataset(FLAGS.data_address,
                              FLAGS.task,
                              mode='train',)

train_loader = DataLoader(train_dataset,
                          batch_size=FLAGS.train_params.batch_size,
                          shuffle=True,
                          collate_fn=collate_multimodal,
                          num_workers=FLAGS.data.num_workers)

val_dataset = QM9DGLDataset(FLAGS.data_address,
                            FLAGS.task,
                            mode='valid')
val_loader = DataLoader(val_dataset,
                        batch_size=FLAGS.train_params.batch_size,
                        shuffle=False,
                        collate_fn=collate_multimodal,
                        num_workers=FLAGS.data.num_workers)

test_dataset = QM9DGLDataset(FLAGS.data_address,
                         FLAGS.task,
                         mode='test')

test_loader = DataLoader(test_dataset,
                         batch_size=FLAGS.train_params.batch_size,
                         shuffle=False,
                         collate_fn=collate_multimodal,
                         num_workers=FLAGS.data.num_workers)

FLAGS.train_size = len(train_dataset)
FLAGS.val_size = len(val_dataset)
FLAGS.test_size = len(test_dataset)

################ 2. Prepare model
model = Chemical_Multimodal_Model(FLAGS.graph_encoder_params, FLAGS.smiles_encoder_params)

if args.pretrained_path is not None:
    model.load_state_dict(torch.load(args.pretrained_path))
model.to(device)

# Optimizer settings
optimizer = optim.Adam(model.parameters(), lr=FLAGS.train_params.init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                           FLAGS.train_params.epochs,
                                                           eta_min=FLAGS.train_params.min_lr)

################ 2. Start training

# Loss function
def task_loss(pred, target, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]

    rescale_loss = train_dataset.norm2units(l1_loss, FLAGS.task)
    return l1_loss, l2_loss, rescale_loss


# Save path
save_path = os.path.join(FLAGS.out_dir, f"{FLAGS.model}_{FLAGS.task}.pt")

# Run training
print('Begin training')
for epoch in range(FLAGS.train_params.epochs):
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")

    train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS, device)
    val_epoch(epoch, model, task_loss, val_loader, FLAGS, device)
    test_epoch(epoch, model, task_loss, test_loader, FLAGS, device)
