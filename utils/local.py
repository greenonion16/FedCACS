import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import DatasetSplit, LocalUpdate
from models.test import test_img_local
from utils.mini_imagenet import get_imagenet_data

import time
import os

if __name__ == '__main__':
    args = args_parser()
    _time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    start_time = time.time()
    print(_time)
    args.time = _time[5:10]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, _ = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'mini_imagenet' in args.dataset:
            dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index = get_imagenet_data(args)
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])

    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    # training
    loss_train = []
    acc_avg = 0

    lr = args.lr
    results = []

    criterion = nn.CrossEntropyLoss()
    indd = None
    accs_ = []
    start = time.time()
    for user in range(min(100, args.num_users)):
        # model_save_path = os.path.join(base_dir, 'local/model_user{}.pt'.format(user))
        net_best = None
        best_acc = None
        net_local = copy.deepcopy(net_glob)

        ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[user]), args.local_bs, shuffle=True)

        optimizer = torch.optim.SGD(net_local.parameters(), lr=lr, momentum=0.5)
        for iter in range(args.epochs):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.cuda(), labels.cuda()
                net_local.zero_grad()
                log_probs = net_local(images)
                labels = labels.long()
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        acc_test, loss_test = test_img_local(net_local, dataset_test, args, user_idx=user,
                                             idxs=dict_users_test[user])
        accs_.append(acc_test)
        print('Average accuracy: {}'.format(sum(accs_) / len(accs_)))
        print('User {}, Loss: {:.2f}, Accuracy: {:.2f}'.format(user, loss_test, acc_test))
        acc_avg += acc_test / args.num_users

        del net_local

    end = time.time()
    print(end - start)
    print(accs_)
    print('Average accuracy: {}'.format(acc_avg))
