# CAGN
This repository is the offical PyTorch implementation of CAGN.
```
@article{xu2025complex,
  title={Complex graph neural networks for multi-hop propagation},
  author={Xu, Dong and Liu, Ziye and Li, Fengming and Meng, Yulong},
  journal={Neurocomputing},
  pages={130364},
  year={2025},
  publisher={Elsevier}
}
```
## Enviroment Setup
The experiments were conducted under this specific environment:

1. Ubuntu 20.04.3 LTS
2. Python 3.8.10
3. CUDA 10.2
4. Torch 1.11.0 (with CUDA 10.2)

In addition, torch-scatter, torch-sparse and torch-geometric are needed to handle scattered graphs and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. For these three packages, follow the official instructions for [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-sparse](https://github.com/rusty1s/pytorch_sparse), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Acknowledgements

The template is borrowed from [MagNet](https://github.com/matthew-hirn/magnet) and Pytorch-Geometric Signed Directed. We thank the authors for the excellent repositories.
