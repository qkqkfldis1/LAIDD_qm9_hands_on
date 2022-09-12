# Introduction
- Graph transformer 활용한 분자 물성 예측 실습입니다. 
- DGL 기반으로 실습하며, QM9 dataset을 활용합니다.

# Installation

- pytorch
- dgl (https://www.dgl.ai/pages/start.html)
    - 쿠다, 파이썬 버젼에 맞게 설치하시면 됩니다.
- SmilesPE
- transformers

# Data preparation
1. https://github.com/FabianFuchsML/se3-transformer-public/tree/master/experiments/qm9 에 있는, qm9 dataset [link](https://drive.google.com/file/d/1EpJG0Bo2RPK30bMKK6IUdsR5r0pTBEP0/view) 를 사용하여 QM9_data.pt 다운로드
2. ./data directory 로 옮기시면 됩니다. 

# running
~~~
python train_transformer.py --config ./configs/graph_transformer.yaml
python train_transformer.py --config ./configs/graph_transformer_edge.yaml
~~~