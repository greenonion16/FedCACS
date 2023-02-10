import copy
import itertools
import numpy as np
import pandas as pd
import torch
import os

from utils.options import args_parser
from utils.train_utils import get_data, get_model, headavg
from utils.mini_imagenet import get_imagenet_data
from models.Update import LocalUpdate, LocalUpdateCACS
from models.test import test_img_local_all, test_img

import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    _time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    start_time = time.time()
    print(_time)
    args.time = _time[5:10]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f'''Starting training:
        Algorithm:              {args.alg}
        Epochs:                 {args.epochs}
        Learning rate:          {args.lr}
        Device:                 {args.gpu}
        Ratio:                  {args.frac}
        Num users:              {args.num_users}
        Local Batch size:       {args.local_bs}
        Local epoch:            {args.local_ep}
        Local rep epoch:        {args.local_second_ep}
        shard_per_user:         {args.shard_per_user}
        multi_cats:             {args.multi_cats == 1}
        user_cat_list:          {args.user_cat_list}
        use_cos_loss:           {args.cos == 1}

    ''')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'fmnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, user_cat_index = get_data(args)
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

    total_num_layers = len(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    global_rep_weights = []
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'fedcacs':
        if 'cifar' in args.dataset:
            if args.model == 'cnn':
                global_rep_weights = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
                global_rep_weights = list(itertools.chain.from_iterable(global_rep_weights))
            elif args.model == 'vgg':
                global_rep_weights = list(net_glob.state_dict().keys() - ['classifier.6.weight', 'classifier.6.bias'])
            else:
                print('weights wrong!')

        elif args.dataset == 'mini_imagenet':
            if args.cos:
                global_rep_weights = net_keys[2:]
            else:
                global_rep_weights = net_keys[:-2]
        elif args.dataset == 'fmnist':
            global_rep_weights = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
        elif args.dataset == 'femnist':
            global_rep_weights = [net_glob.weight_keys[i] for i in [0, 1]]
        elif 'sent140' in args.dataset:
            global_rep_weights = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            global_rep_weights = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            global_rep_weights = [net_glob.weight_keys[i] for i in [1, 2]]
            global_rep_weights = list(itertools.chain.from_iterable(global_rep_weights))
        elif args.dataset == 'mini_imagenet':
            if args.cos:
                global_rep_weights = net_keys[2:]
            else:
                global_rep_weights = net_keys[:-2]
        else:
            global_rep_weights = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        global_rep_weights = []

    head_weights_keys = list(set(net_keys) - set(global_rep_weights))
    print('Total_num_layers:', total_num_layers, '\n')
    print('Global_rep_weights:', global_rep_weights, '\n')
    print('net_keys:', net_keys, '\n')
    print('head_weights_keys', head_weights_keys)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            if key in global_rep_weights:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    local_nets_dict = {}
    for user in range(args.num_users):
        net_weights = {}
        for key in net_keys:
            net_weights[key] = copy.deepcopy(net_glob.state_dict()[key])
        local_nets_dict[user] = net_weights

    indd = None
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    for iter in range(args.epochs + 1):
        epoch_temp_net_weights = {}
        loss_locals = []

        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        times_in = []
        total_len = 0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if args.epochs == iter:
                if args.alg == 'fedrep' or 'lg' or 'fedper':
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
                elif args.alg == 'fedcacs':
                    local = LocalUpdateCACS(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
                else:
                    exit('Error: unrecognized localupdate')
            else:
                if args.alg == 'fedrep' or 'lg' or 'fedper':
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
                elif args.alg == 'fedcacs':
                    local = LocalUpdateCACS(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
                else:
                    exit('Error: unrecognized localupdate')

            net_local = copy.deepcopy(net_glob)
            net_local_weights = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in local_nets_dict[idx].keys():
                    if k in head_weights_keys:
                        net_local_weights[k] = copy.deepcopy(local_nets_dict[idx][k])
            net_local.load_state_dict(net_local_weights)

            last = iter == args.epochs

            updated_net_local_weights, loss, indd = local.train(net=net_local.cuda(), idx=idx,
                                                                    w_glob_keys=global_rep_weights,
                                                                    lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]

            if len(epoch_temp_net_weights) == 0:
                epoch_temp_net_weights = copy.deepcopy(updated_net_local_weights)
                for key in net_keys:
                    local_nets_dict[idx][key] = copy.deepcopy(updated_net_local_weights[key])
            else:
                for key in net_keys:
                    epoch_temp_net_weights[key] += copy.deepcopy(updated_net_local_weights[key])
                    local_nets_dict[idx][key] = copy.deepcopy(updated_net_local_weights[key])

            times_in.append(time.time() - start_in)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        for key in net_keys:
            epoch_temp_net_weights[key] = torch.div(epoch_temp_net_weights[key], total_len)
        net_weights_after_aggregation = copy.deepcopy(epoch_temp_net_weights)

        if args.alg == 'fedrep' or args.alg == 'fedcacs' or args.alg == 'fedper':
            if iter > args.epochs - 4:
                if iter == args.epochs - 3:
                    print('-------------head_avg--------------')
                glob_head = headavg(args, local_nets_dict, head_weights_keys, user_cat_index)
                wo_novel_length = args.num_classes-args.novel_class_num
                for key in head_weights_keys:
                    if 'bias' not in key:
                        if args.cos:
                            net_weights_after_aggregation[key][:, :wo_novel_length] = glob_head
                        else:
                            net_weights_after_aggregation[key][:wo_novel_length, :] = glob_head

        net_glob_weights = net_glob.state_dict()
        for key in net_keys:
            net_glob_weights[key] = net_weights_after_aggregation[key]
        if args.epochs != iter:
            net_glob.load_state_dict(net_glob_weights)

        if (iter + 1) % args.test_freq == 0:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            all_local_cor, all_local_loss = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                               w_glob_keys=global_rep_weights, w_locals=local_nets_dict,
                                                               indd=indd,
                                                               dataset_train=dataset_train,
                                                               dict_users_train=dict_users_train,
                                                               return_all=False)

            glob_loss, glob_cor = test_img(net_weights_after_aggregation, args, dataset_test)

            accs.append(all_local_cor)
            print('Round {:3d}, Train loss: {:.3f}, glob_loss: {:.3f}, glob_cor: {:.2f}, all_local_loss: {:.2f}, '
                  'all_local_cor: {:.2f}'
                  .format(iter, loss_avg, glob_loss, glob_cor, all_local_loss, all_local_cor))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += all_local_cor / 10

        if iter == args.epochs:
            print('******save model******')
            model_save_path = './save/5-10/' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + 'users' + '_' + str(
                args.user_cat_list) + '_all_local_cor' + str(all_local_cor)[:5] + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
