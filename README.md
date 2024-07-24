# LBCS
This is the code for ICML 2024 paper: [Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints](https://openreview.net/pdf?id=yb5xV8LFDq).

# Abstract
Coreset selection is powerful in reducing computational costs and accelerating data processing for deep learning algorithms. It strives to identify a small subset from large-scale data, so that training only on the subset practically performs on par with full data. Practitioners regularly desire to identify the smallest possible coreset in realistic scenes while maintaining comparable model performance, to minimize costs and maximize acceleration. Motivated by this desideratum, for the first time, we pose the problem of refined coreset selection, in which the minimal coreset size under model performance constraints is explored. Moreover, to address this problem, we propose an innovative method, which maintains optimization priority order over the model performance and coreset size, and efficiently optimizes them in the coreset selection procedure. Theoretically, we provide the convergence guarantee of the proposed method. Empirically, extensive experiments confirm its superiority compared with previous strategies, often yielding better model performance with smaller coreset sizes.

# Setup
First, install dependencies.

```bash
pip install -r requirements.txt
```

Download all the data into `dataset` folder.

# Experiments
Experiments on SVHN.

```bash
python run_svhn.py --init 1000 --limit 1200
python run_svhn.py --init 2000 --limit 2200
python run_svhn.py --init 3000 --limit 3200
python run_svhn.py --init 4000 --limit 4200
```

Experiments on F-MNIST.

```bash
python run_fashion.py --init 1000 --limit 1200
python run_fashion.py --init 2000 --limit 2200
python run_fashion.py --init 3000 --limit 3200
python run_fashion.py --init 4000 --limit 4200
```

Experiments on CIFAR-10.

```bash
# Obatin a list of checkpoints for the inner loop
python get_checkpoint.py

python run_cifar.py --init 1000 --limit 1200
python run_cifar.py --init 2000 --limit 2200
python run_cifar.py --init 3000 --limit 3200
python run_cifar.py --init 4000 --limit 4200

```

Experiments on F-MNIST with noisy labels.

```bash
python noisify_fashion.py

python run_fashion.py --init 1000 --limit 1200 --noisify
python run_fashion.py --init 2000 --limit 2200 --noisify
python run_fashion.py --init 3000 --limit 3200 --noisify
python run_fashion.py --init 4000 --limit 4200 --noisify
```

Experiments on Long-tailed F-MNIST.

```bash
python imbalance_fashion.py

python run_fashion.py --imbalance --init 1000 --limit 1200 --dataset_size 14886
python run_fashion.py --imbalance --init 2000 --limit 2200 --dataset_size 14886
python run_fashion.py --imbalance --init 3000 --limit 3200 --dataset_size 14886
python run_fashion.py --imbalance --init 4000 --limit 4200 --dataset_size 14886
```

# Reference
If you find the paper and code useful, please cite our paper.

```
@inproceedings{xia2024refined,
title={Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints},
author={Xiaobo Xia and Jiale Liu and Shaokun Zhang and Qingyun Wu and Hongxin Wei and Tongliang Liu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=yb5xV8LFDq}
}
```
