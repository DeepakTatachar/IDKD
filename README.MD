# In-Distribution Knowledge Distillation (IDKD)

This is the official implementation for the paper "Homogenizing Non-IID datasets via In-Distribution Knowledge Distillation for Decentralized Learning"
Accepted at Transactions on Machine Learning Research (TMLR) [paper link](https://openreview.net/pdf?id=CuyJkNjIVd), [video](https://youtu.be/I-den-2TNyk)

If you use this code please consider citing
```bibtex
@article{
  ravikumar2024homogenizing,
  title={Homogenizing Non-{IID} Datasets via In-Distribution Knowledge Distillation for Decentralized Learning},
  author={Deepak Ravikumar and Gobinda Saha and Sai Aparna Aketi and Kaushik Roy},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=CuyJkNjIVd},
}
```

## Requirements
```
- pytorch 1.7.1
- torchvision 0.8.2
- mpich 3.2
- numpy 1.18.4
```

## Setup
Update the paths for the datasets in  ./utils/load_dataset.py, replace the string 'Set path'

Create a folder "./pretrained/\<dataset name\>" and "./pretrained/\<dataset name\>/temp"
i.e. 
```
mkdir pretrained
mkdir pretrained/cifar10
mkdir pretrained/cifar10/temp
```

Output of each node is logged into "./logs" folder with a corresponding log file.

## Train with IDKD framework 

To train a 16 node ring network run, set the Dirichlet parameter alpha an training dataset set
```
mpirun -np 16 python train_idkd.py --dataset=cifar10 --alpha=0.05
```

Graph configuration can be set by using the network argument
```
mpirun -np 16 python train_idkd.py --dataset=cifar10 --alpha=0.05 --network=ring
```

Supported network types are 'ring' and 'social15'

**Training DSGDm use 16 node ring network**

```
mpirun -np 16 python train_dpsgd.py --dataset=cifar10 --alpha=0.05 --network=ring
```

**Training QG-DSGDm-N use 16 node ring network**

```
mpirun -np 16 python train_qgm.py --dataset=cifar10 --alpha=0.05 --network=ring
```

**Training Relay-Sum SGD use 16 node ring network**
```
mpirun -np 16 python train_relay_sgd.py --dataset=cifar10 --alpha=0.05
```

To run mpi on multiple hosts use the host argument followed by a list of hostnames, for example
```
mpirun -np 16 --hosts <host_names> python train_relay_sgd.py --dataset=cifar10 --alpha=0.05
```

Dataset | Param |
------------- |:-------------:| 
CIFAR-10 | --dataset=cifar10 |
CIFAR-100 | --dataset=cifar100 |
ImageNette | --dataset=imagenette |
