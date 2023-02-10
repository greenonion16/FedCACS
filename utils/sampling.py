#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch
from tqdm import trange

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all


def new_noniid(dataset, num_users, user_cat_list, num_classes, novel_class_num, user_cat_index=None):
    if user_cat_index is None:
        user_cat_index = []
    np.random.seed(2)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        try:
            label = torch.tensor(dataset.targets[i]).item()
        except:
            label = torch.tensor(dataset.labels[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    # 生成类别数列表，cat_num_list:[37，43，20]
    cat_num_list = [100]
    # cat_num_list = []
    # now_num = num_users
    # for _ in range(len(user_cat_list)-1):
    #     cat_num_list.append(np.random.choice(now_num))
    #     now_num -= cat_num_list[-1]
    # cat_num_list.append(now_num)

    # 生成user_cat_index，即保存每个user上拥有的类别的list
    num_classes -= novel_class_num

    cat_remainder = int(np.inner(cat_num_list, user_cat_list)) % num_classes
    num_class_list = [np.int(cat) for cat in range(num_classes)]
    cat_plus1 = list(np.random.choice(num_class_list, cat_remainder, replace=False))
    num_class_list = num_class_list * (int(np.inner(cat_num_list, user_cat_list)) // num_classes) + cat_plus1
    if not user_cat_index:
        for idx, per_cat_num in enumerate(cat_num_list):
            cat_num = user_cat_list[idx]
            for user in range(per_cat_num):
                user_cat = np.random.choice(len(num_class_list), cat_num, replace=False)
                cats = [num_class_list[i] for i in user_cat]
                # while len(cats) != len(set(cats)):
                #     user_cat = np.random.choice(len(num_class_list), cat_num, replace=False)
                #     cats = [num_class_list[i] for i in user_cat]
                user_cat_index.append(cats)
                num_class_list = np.delete(num_class_list, user_cat)

    for label in idxs_dict.keys():
        cat_div = int(np.inner(cat_num_list, user_cat_list)) // num_classes
        if int(label) in cat_plus1:
            cat_div += 1
        x = idxs_dict[label]
        img_leftover = len(x) % cat_div
        leftover = x[-img_leftover:] if img_leftover > 0 else []
        x = np.array(x[:-img_leftover]) if img_leftover > 0 else np.array(x)
        x = x.reshape((cat_div, -1))
        x = list(x)
        x[0] = np.concatenate((x[0], leftover))
        idxs_dict[label] = x

    # divide and assign
    for i in trange(num_users):
        rand_set_label = user_cat_index[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    return dict_users, user_cat_index


def data_idx_dict(args, dataset):
    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < args.num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < args.num_classes:
            idxs_dict[label].append(i)
            count += 1
    return idxs_dict


def novel_data(args, dataset_train, dataset_test):
    novel_user = args.novel_users
    n_ways = 5
    shot_num = 100

    train_idxs_dict = data_idx_dict(args, dataset_train)
    test_idxs_dict = data_idx_dict(args, dataset_test)

    novel_classes = [class_num for class_num in range(args.num_classes-args.novel_class_num, args.num_classes)]
    # novel_classes = [class_num for class_num in range(10)]

    train_dict_users = {i: np.array([], dtype='int64') for i in range(novel_user)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(novel_user)}
    novel_user_cat_index = []
    for user in range(novel_user):
        novel_user_cat_index.append(random.sample(novel_classes, n_ways))

    # novel_user_cat_index for example:
    #    [[98, 96, 97, 92, 91],
    #      [94, 92, 93, 95, 96],
    #      [98, 93, 99, 90, 96],
    #      [99, 91, 90, 95, 92],
    #      [96, 95, 98, 91, 90],
    #      [91, 92, 97, 98, 95],
    #      [93, 99, 94, 95, 96],
    #      [95, 94, 92, 90, 97],
    #      [96, 93, 90, 97, 92],
    #      [90, 99, 94, 96, 92]]

    for idx, one_user_cat_list in enumerate(novel_user_cat_index):
        train_img_ind = np.array([], dtype='int64')
        test_img_ind = np.array([], dtype='int64')
        for cat_num in one_user_cat_list:
            train_img = np.random.choice(train_idxs_dict[cat_num], shot_num, replace=False)
            train_img_ind = np.concatenate(
                (train_img_ind, train_img))

            # test_img_ind = np.concatenate(
            #     (test_img_ind, np.random.choice(test_idxs_dict[cat_num], len(test_idxs_dict[cat_num]), replace=False)))

            test_img_ind = np.concatenate(
                (test_img_ind, np.array(test_idxs_dict[cat_num])))

        train_dict_users[idx] = train_img_ind
        test_dict_users[idx] = test_img_ind

    # train_dict_users:
    # {0: array([33067, 18532, 42858, 1451, 42763, 10713, 35879, 12291, 6692,
    #            38106, 15034, 40383, 43123, 24354, 15588, 18212, 39740, 45488,
    #            32142, 49980, 15160, 46032, 3184, 28449, 20653, 36318, 667,
    #            47476, 17769, 31562, 30463, 34988, 5579, 14502, 7793, 15216,
    #            31610, 44800, 27512, 17158, 34976, 24838, 40657, 45927, 10960,
    #            5222, 1649, 5925, 33154, 4323]),
    #  .....
    #  9: array([13881, 36866, 37541, 38928, 22091, 9325, 45406, 20341, 38222,
    #            30822, 33445, 34620, 22615, 1623, 31313, 8623, 4370, 39005,
    #            25810, 18234, 25470, 33055, 5266, 25452, 36572, 33825, 36779,
    #            11146, 45159, 28605, 8071, 23030, 12337, 35063, 25703, 33380,
    #            9978, 42997, 42776, 34894, 49436, 42354, 12001, 32663, 48054,
    #            24452, 35778, 18239, 48221, 40126])}

    return train_dict_users, test_dict_users, novel_user_cat_index



