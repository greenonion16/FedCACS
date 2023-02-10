# Local and Global Bilaterally Improvement in Federated Learning


## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies: `pip3 install -r requirements.txt`.

## Data

This code uses the CIFAR10, CIFAR100, mini-ImageNet datasets.

The CIFAR10, CIFAR100 AND MNIST datasets are downloaded automatically by the torchvision package. 
The mini-ImageNet dataset can be downloaded from this link: https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE.


## Usage

FedCACS is run using a command of the following form:

`python main.py --alg fedcacs --dataset [dataset] --num_users [num_users] --model [model] --model [model] --shard_per_user [shard_per_user] --epochs [model]
Explanation of parameters:

- `alg` : algorithm to run, may be `fedcacs`, `fedrep` (FedRep), `fedavg`, `fedper` (FedPer), or `lg` (LG-FedAvg)
- `dataset` : dataset, may be `cifar10`, `cifar100`, `mini_imagenet`
- `num_users` : number of users
- `model` : for the CIFAR datasets, we use `cnn`, for the MNIST datasets, we use `mlp`
- `num_classes` : total number of classes
- `shard_per_user` : number of classes per user (specific to CIFAR datasets and MNIST)
- `frac` : fraction of participating users in each round (for all experiments we use 0.1)
- `local_bs` : batch size used locally by each user
- `lr` : learning rate
- `epochs` : total number of communication rounds
- `local_ep` : total number of local epochs
- `rep_eps` : number of local epochs to execute for the representation (specific to FedCACS and FedRep)
- `gpu` : GPU ID

A full list of configuration parameters and their descriptions are given in `utils/options.py`.

