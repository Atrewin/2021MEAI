import os
import numpy as np
import sys
import torch
from torch import autograd, optim, nn

from transformers import AdamW, get_linear_schedule_with_warmup
import traceback
from utils.logger import *
from utils.json_util import *

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0




class AdeversarialTrainingFramework:

    def __init__(self, model, source_dataloader, target_dataloader, class_criterion, domain_criterion, optimizer, source_classes, target_classes, opt):
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
        self.opt = opt
        self.source_classes = source_classes
        self.target_classes = target_classes
        # self.source_dataset_val = source_dataset_val


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
        self.model.train()
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0
        total_hit_d, total_num_d = 0.0, 0.0

        for i, ((source_data, source_label), (target_data, _)) in enumerate( zip(source_dataloader, target_dataloader)):
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

            # 够着领域标签
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).long().cuda()
            # 設定source data的label为1
            domain_label[:source_data.shape[0]] = 1
            domain_label = domain_label.reshape(-1)
            class_logits, domain_logits = self.model(source_data, target_data, alpha=0.2)
            class_logits = class_logits[:, :self.source_classes]
            # class_logits =
            DA_loss = self.class_criterion(domain_logits, domain_label)
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
            predict = torch.argmax(class_logits, dim=1)
            hit = torch.sum( predict == source_label).item()
            total_hit += hit
            total_num += source_data.shape[0]
            total_hit_d += torch.sum(torch.argmax(domain_logits, dim=1) == domain_label).item()
            total_num_d += domain_logits.shape[0]



            sys.stdout.write('step: {0:4} | source SA loss: {1:2.6f}, source SA accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                        .format(i + 1,
                                CA_loss,
                                100 * total_hit / total_num,
                                DA_loss,
                                100 * total_hit_d / total_num_d,
                                ) + '\r')

            sys.stdout.flush()

        # logger.info(
        #     'step: {0:4} | source SA loss: {1:2.6f}, source SA accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
        #     .format(i + 1,
        #             CA_loss,
        #             100 * total_hit / total_num,
        #             DA_loss,
        #             100 * total_hit_d / total_num_d,
        #             ) + '\r')

        return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num * 100

    def train(self, epoch=30):
        if(self.opt.ckpt_name != ''):
            self.model = self.__initi_model__(model=self.model,ckpt=self.opt.ckpt_name)
            pass

        pre_train_acc = 0
        view_s = {
            "source_acc":[],
            "target_acc":[],
            "epoch":[]

        }
        for epoch in range(epoch):

            train_D_loss, train_F_loss, train_acc = self.train_epoch(self.source_dataloader, self.target_dataloader)
            if pre_train_acc < train_acc:
                # torch.save(self.model.extractor_model.state_dict(), f'extractor_model.bin')
                # torch.save(self.model.label_predictor.state_dict(), f'predictor_model.bin')
                path = "output/" + str(self.source_classes) + "-" + str(self.target_classes) + "-" + self.opt.notes + "-" + 'model.bin'
                torch.save(self.model.state_dict(), path)

            pre_train_acc = train_acc

            target_acc = self.target_inference(self.model)

            view_s["source_acc"].append(train_acc)
            view_s["target_acc"].append(target_acc)
            view_s["epoch"].append(epoch)

            if(epoch%10):
                path = "output/" + str(self.source_classes) + "-" + str(self.target_classes) + "-" + self.opt.notes + "-acc.json"
                dump(view_s , path)

            logger.info(
                'epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, source_acc {:6.4f}, target_acc {:6.4f}'.format(epoch, train_D_loss,
                                                                                                 train_F_loss,
                                                                                                 train_acc,
                                                                                                 target_acc))
        name = "checkpoint is saved in " + str(self.source_classes) + "-" + str(self.target_classes) + "-" + self.opt.notes + f'model.bin'
        logger.info("Complete training")
        logger.info(name)
        # path = str(self.source_classes) + "" + str(self.target_classes) + "-" + self.opt.notes + 'model.bin'
    def target_inference(self, model, ckpt=None):

        model.eval()
        # if ckpt is None:
        #     pass
        # else:
        #     if ckpt != 'none':
        #         state_dict = self.__load_model__(ckpt)
        #         own_state = model.state_dict()
        #         for name, param in state_dict.items():
        #             if name not in own_state:
        #                 continue
        #             own_state[name].copy_(param)
        model = self.__initi_model__(model=model,ckpt=ckpt)
        model.eval()
        result = []

        total_hit = 0
        total = 0
        for i, (test_data, label) in enumerate(self.target_dataloader):
            test_data = test_data.cuda()
            label = label.cuda()
            class_logits = model.target_domain_predict(test_data)
            class_logits = class_logits[:, :self.target_classes]

            predict = torch.argmax(class_logits, dim=1)
            hit = torch.sum(predict == label).item()

            predict = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
            result.append(predict)
            total_hit += hit
            total += len(predict)

        return  total_hit/total * 100
        # import pandas as pd
        # result = np.concatenate(result)

        # # Generate your submission
        # df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
        # df.to_csv('DANN_target_domain.csv', index=False)

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

    def __initi_model__(self,model, ckpt):#确保传引用

        if ckpt is None:
            pass
        else:
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)

        return model

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
