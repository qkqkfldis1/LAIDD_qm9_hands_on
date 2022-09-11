# Introduction
- Graph transformer 활용한 분자 물성 예측 실습입니다. 
- DGL 기반으로 실습하며, QM9 dataset을 활용합니다.

# Installation

- pytorch
- dgl (https://www.dgl.ai/pages/start.html)
    - 쿠다, 파이썬 버젼에 맞게 설치하시면 됩니다.
- SmilesPE
- transformers

# running
~~~
python train_transformer.py --config ./configs/graph_transformer.yaml
python train_transformer.py --config ./configs/graph_transformer_edge.yaml
~~~