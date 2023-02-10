import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[np.int(self.idxs[item])]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


class finetuneLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=10, shuffle=True)

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, lr=0.1):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': net.weight_base, 'lr': 0.5},
                # {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        epoch_loss = []
        num_updates = 0
        for iter in range(self.args.finetune_epochs):

            # all other methods update all parameters simultaneously
            # for name, param in net.named_parameters():
            #     param.requires_grad = True

            for name, param in net.named_parameters():
                if name in [net.weight_keys[i] for i in [0, 1, 3, 4]]:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            # feature_list = torch.tensor([]).cuda()
            # cate_list = torch.tensor([]).cuda()

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(batch_idx, labels)
                images, labels = images.cuda(), labels.cuda()
                net.zero_grad()
                log_probs = net(images)
                # if iter == self.args.finetune_epochs-1:
                #     feature = torch.div(torch.sum(feature, dim=0), 10).expand(1, -1)
                #     feature_list = torch.cat((feature_list, feature), dim=0)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())


            # prototype = torch.mean(feature_list, dim=0)
            # print(prototype)

            # epoch_loss.append(sum(batch_loss) / len(batch_loss))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net, sum(epoch_loss) / len(epoch_loss)
