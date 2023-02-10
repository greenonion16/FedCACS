import copy
import numpy as np
import pandas as pd
import torch
import os
import time

from utils.options import args_parser
from utils.train_utils import get_novel_data, get_model, get_data
from models.novelUpdate import finetuneLocalUpdate
from models.test import test_novel, test_img
from tqdm import trange

args = args_parser()


def load_model():
    net_glob = get_model(args)
    net_glob.train()

    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    return net_glob


def replace_classifier(net, head_keys):
    new_net = copy.deepcopy(net)
    novel_net_weights = new_net.state_dict()

    cls_weights = novel_net_weights[head_keys]
    real_cls_weights = cls_weights[:, :args.num_classes - args.novel_class_num]
    avg_cls_weight = torch.div(torch.sum(real_cls_weights, dim=1), real_cls_weights.size(dim=1))
    avg_cls_weight = avg_cls_weight.unsqueeze(0).transpose(0, 1).repeat(1, args.novel_class_num)
    cls_weights[:, -args.novel_class_num:] = avg_cls_weight

    new_net.load_state_dict(novel_net_weights)

    return new_net


def replace_cla(net, head_keys, cat_index, proto, user_num):
    new_net = copy.deepcopy(net)
    novel_net_weights = new_net.state_dict()

    cls_weights = novel_net_weights[head_keys]

    user_cate_list = cat_index[user]
    for idx, cate_num in enumerate(user_cate_list):
        print(cate_num)
        cls_weights[:, cate_num] = proto[idx]

    new_net.load_state_dict(novel_net_weights)

    return new_net


if __name__ == '__main__':
    _time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    start_time = time.time()
    print(_time)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    dataset_train, dataset_test, dict_users_train, dict_users_test, novel_user_cat_index = get_novel_data(args)

    pre_net = load_model()
    net_head_keys = pre_net.weight_keys[2][0]
    novel_net = replace_classifier(pre_net, net_head_keys)
    novel_net.train()

    pre_acc, pre_loss = [], []
    novel_acc, novel_loss = [], []

    for user in trange(args.novel_users):
        pre_updater = finetuneLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[user])
        novel_updater = finetuneLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[user])

        pre_net, loss_1 = pre_updater.train(net=pre_net.cuda(), lr=args.lr)
        novel_net, loss_2 = novel_updater.train(net=novel_net.cuda(), lr=args.lr)

        user_pre_loss, user_pre_cor = test_novel(args, pre_net, dataset_test, idxs=dict_users_test[user])
        user_novel_loss, user_novel_cor = test_novel(args, novel_net, dataset_test, idxs=dict_users_test[user])

        pre_loss.append(copy.deepcopy(user_pre_loss))
        novel_loss.append(copy.deepcopy(user_novel_loss))
        pre_acc.append(copy.deepcopy(user_pre_cor))
        novel_acc.append(copy.deepcopy(user_novel_cor))

    pre_loss = sum(pre_loss) / len(pre_loss)
    novel_loss = sum(novel_loss) / len(novel_loss)
    pre_acc = sum(pre_acc) / len(pre_acc)
    novel_acc = sum(novel_acc) / len(novel_acc)

    print('pre_loss: {:.3f}, novel_loss: {:.3f}, pre_cor: {:.2f}, novel_cor: {:.2f}'
          .format(pre_loss, novel_loss, pre_acc, novel_acc))
