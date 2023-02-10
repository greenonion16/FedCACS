import copy

from torchvision import datasets, transforms
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP, CNNCifar_PFL, CNN_FEMNIST, get_vgg11
from models.new_Nets import cos_cifar100, cos_cifar10, ConvNet, ConvNet_wocos
from utils.sampling import noniid, new_noniid, novel_data
import os
import json
import torch

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.multi_cats:
            dict_users_train, user_cat_index = new_noniid(dataset_train, args.num_users, args.user_cat_list,
                                                          args.num_classes, args.novel_class_num)
            dict_users_test, user_cat_index = new_noniid(dataset_test, args.num_users, args.user_cat_list,
                                                         args.num_classes, args.novel_class_num, user_cat_index=user_cat_index)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,
                                                    args.num_classes)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                   rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        if args.multi_cats:
            dict_users_train, user_cat_index = new_noniid(dataset_train, args.num_users, args.user_cat_list,
                                                          args.num_classes, args.novel_class_num)
            dict_users_test, user_cat_index = new_noniid(dataset_test, args.num_users, args.user_cat_list,
                                                         args.num_classes, args.novel_class_num, user_cat_index=user_cat_index)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,
                                                    args.num_classes)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                   rand_set_all=rand_set_all)
    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('data/fmnist', train=True, download=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize((32, 32)),
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,)),
                                              ]))

        # testing
        dataset_test = datasets.FashionMNIST('data/fmnist/', train=False, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index


def get_novel_data(args):
    if args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        train_dict_users, test_dict_users, novel_user_cat_index = novel_data(args, dataset_train, dataset_test)
    else:
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        train_dict_users, test_dict_users = None, None

    return dataset_train, dataset_test, train_dict_users, test_dict_users, novel_user_cat_index


def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        if args.cos:
            print('use cos_loss')
            net_glob = cos_cifar100(args=args).cuda()
        else:
            print('use usual_loss')
            net_glob = CNNCifar100(args=args).cuda()
    elif args.model == 'vgg' and 'cifar100' in args.dataset:
        net_glob = get_vgg11(100).cuda()
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        if args.cos:
            print('use cos_loss')
            net_glob = cos_cifar10(args=args).cuda()
        else:
            print('use usual_loss')
            net_glob = CNNCifar(args=args).cuda()
    elif args.dataset == 'mini_imagenet':
        opt = {'userelu': False, 'in_planes': 3, 'out_planes': [64, 64, 64, 64], 'num_stages': 4}
        if args.cos:
            net_glob = ConvNet(opt).cuda()
        else:
            net_glob = ConvNet_wocos(opt).cuda()

    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNNCifar_PFL(args=args).cuda()
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        print('model:MLP')
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).cuda()
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        print('model:CNN_FEMNIST')
        net_glob = CNN_FEMNIST(args=args).cuda()
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).cuda()
    else:
        exit('Error: unrecognized model')

    return net_glob


def headavg(args, local_nets_dict, head_weights_keys, user_cat_index):
    dict_length = args.num_classes-args.novel_class_num
    cate_weight_component_dict = dict(zip([i for i in range(dict_length)], [[] for _ in range(dict_length)]))
    for idx in range(args.num_users):
        cate_set = list(set(user_cat_index[idx]))
        for key in head_weights_keys:
            for line in cate_set:
                if 'bias' not in key:
                    if args.cos:
                        cate_weight_component_dict[line].append(
                            copy.deepcopy(local_nets_dict[idx][key])[:, line].expand(1, -1))
                    else:
                        cate_weight_component_dict[line].append(
                            copy.deepcopy(local_nets_dict[idx][key])[line, :].expand(1, -1))


    glob_head = None
    for idx in range(dict_length):
        cate_tensor = torch.cat(cate_weight_component_dict[idx], dim=0)
        cate_avg_weight = torch.div(torch.sum(cate_tensor, dim=0), cate_tensor.size(dim=0)).expand(1, -1)
        if glob_head is None:
            glob_head = cate_avg_weight
        else:
            glob_head = torch.cat((glob_head, cate_avg_weight), dim=0)
    if args.cos:
        glob_head = glob_head.transpose(0, 1)
    else:
        pass
    return glob_head
