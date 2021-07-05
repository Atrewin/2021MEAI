import os
import numpy as np
import sys
import torch
from torch import autograd, optim, nn

from transformers import AdamW, get_linear_schedule_with_warmup
import traceback
from utils.logger import *

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0




class AdeversarialTrainingFramework:

    def __init__(self, model, source_dataloader, target_dataloader, class_criterion, domain_criterion, optimizer, opt):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        # 准备组件练数据
        self.model=model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.class_criterion = class_criterion
        self.domain_criterion = domain_criterion
        self.optimizer = optimizer
        self.opt =opt


    def train_epoch(self, source_dataloader, target_dataloader, lamb=1):
        '''
          Args:
            source_dataloader: source data的dataloader
            target_dataloader: target data的dataloader
            lamb: 調控adversarial的loss係數。
        '''

        # D loss: Domain Classifier的loss
        # F loss: Feature Extrator & Label Predictor的loss
        # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0

        for i, ((source_data, source_label), (target_data, _)) in enumerate( zip(source_dataloader, target_dataloader)):
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

            # 够着领域标签
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
            # 設定source data的label为0
            domain_label[:source_data.shape[0]] = 0

            class_logits, domain_logits = self.model( source_data, target_data)


            DA_loss = self.domain_criterion(domain_logits, domain_label)
            running_D_loss += DA_loss.item()

            # Step 2 : 訓練Feature Extractor和label_predictor

            CA_loss = self.class_criterion(class_logits, source_label)
            running_F_loss += CA_loss.item()

            # 合并两个loss
            loss = DA_loss + DA_loss
            loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            # 我觉得应该发生在loss.backward前面

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]
            print(i, end='\r')

        return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num

    def train(self, epoch=30):
        # 訓練200 epochs
        for epoch in range(epoch):
            train_D_loss, train_F_loss, train_acc = self.train_epoch(self.source_dataloader, self.target_dataloader)

            torch.save(self.model.extractor_model.state_dict(), f'extractor_model.bin')
            torch.save(self.model.label_predictor.state_dict(), f'predictor_model.bin')

            print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch,
                                                                                                   train_D_loss,
                                                                                                    train_F_loss,
                                                                                                       train_acc))

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))#0.2500,
