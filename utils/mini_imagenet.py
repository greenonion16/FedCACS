import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from utils.sampling import new_noniid, noniid

_MINI_IMAGENET_DATASET_DIR = './data/MiniImagenet'


def load_data(file):
    with open(file, 'rb') as fo:
        data_load = pickle.load(fo, encoding='bytes')
    return data_load


def buildLabelIndex(labels):
    idxs_dict = {}
    for i in range(len(labels)):
        label = torch.tensor(labels[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    return idxs_dict

    # test_phase_2 = os.path.join(_MINI_IMAGENET_DATASET_DIR, 'miniImageNet_category_split_test.pickle')
    # data_test_2 = load_data(test_phase_2)


class MiniImageNet(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        self.base_folder = 'miniImagenet'
        assert (phase == 'train' or phase == 'test')
        self.phase = phase
        self.name = 'MiniImageNet_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        train_phase = os.path.join(_MINI_IMAGENET_DATASET_DIR, 'miniImageNet_category_split_train_phase_train.pickle')
        test_phase_1 = os.path.join(_MINI_IMAGENET_DATASET_DIR, 'miniImageNet_category_split_train_phase_test.pickle')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(train_phase)
            self.data = data_train[b'data']
            self.labels = data_train[b'labels']

            self.dict = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.dict.keys())
            self.num_cats = len(self.labelIds)

        elif self.phase == 'test':
            data_test = load_data(test_phase_1)

            self.data = data_test[b'data']
            self.labels = data_test[b'labels']

            self.dict = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.dict.keys())
            self.num_cats = len(self.labelIds)

        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase == 'test') or (do_not_use_random_transf == True):
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def get_imagenet_data(args):
    dataset_train = MiniImageNet(phase='train')
    dataset_test = MiniImageNet(phase='test')
    user_cat_index = None
    if args.multi_cats:
        dict_users_train, user_cat_index = new_noniid(dataset_train, args.num_users, args.user_cat_list,
                                                      args.num_classes, args.novel_class_num)
        dict_users_test, user_cat_index = new_noniid(dataset_test, args.num_users, args.user_cat_list,
                                                     args.num_classes, args.novel_class_num,
                                                     user_cat_index=user_cat_index)
    else:
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user,
                                                args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all)

    return dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index
