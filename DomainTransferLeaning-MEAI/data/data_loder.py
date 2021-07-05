import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from utils.path_util import from_project_root

class DataConfig():#用于对齐两个领域的数据集label to index
    def __init__(self):
        self.label2index_maps = {
            "cr":0,
            "in":1,
            "pa":2,
            "ps":3,
            "rs":4,
            "sc":5,
            "gg":6,
            "rp":7,
            "sp":8
        }
        self.index2label_maps = { self.label2index_maps[key]:key for key in self.label2index_maps.keys()}

    def set_data_config_classes(self, classes):
        self.data_classes = classes

    def label_transform(self, index: int):
        label = self.data_classes[index]
        return self.label2index_maps[label]


source_transform = transforms.Compose([
    # 灰度值
    transforms.Grayscale(),
    # 水平反转 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋转15度內 (Augmentation)
    transforms.RandomRotation(15),
    # to toTensor。
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    # 灰度值。
    transforms.Grayscale(),
    # source data是64x46，将target data的28x28放大成64x64。
    transforms.Resize((64, 64)),
    # 水平反转 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋转15度內 (Augmentation)
    transforms.RandomRotation(15),
    # to Tensor
    transforms.ToTensor(),
])
# # 之后需要加入from root 的设计

# dataConfig = DataConfig()
# source_dataset = ImageFolder('datasets/train_val/train', transform=source_transform, target_transform =dataConfig.label_transform)
# #修改dataConfig的in_classer2str
# dataConfig.set_data_config_classes(source_dataset.classes)
# target_dataset = ImageFolder('datasets/NEU-CLS', transform=target_transform)
#
# a0 = source_dataset[0][0]
# a1 = source_dataset[0][0]
print(" ")


# class DeepModelDataset(data.Dataset):
#     """
#         Deep model Dataset
#         return
#         """
#
#     def __init__(self, root='data/datasets/train_val/train'):
#         self.root = from_project_root(root)
#
#         if not os.path.exists(root):
#             print("[ERROR] Data file does not exist!")
#             assert (0)
#         self.images_dataset = ImageFolder(self.root, transform=source_transform)
#
#         return self.images_dataset


def get_deep_dataset(root):
    root = from_project_root(root)

    if not os.path.exists(root):
        print("[ERROR] Data file does not exist!")
        assert (0)
    # 声明该数据集 index2lable的config
    dataConfig = DataConfig()
    images_dataset = ImageFolder(root, transform=source_transform, target_transform =dataConfig.label_transform)
    #修改dataConfig的in_classer2str
    dataConfig.set_data_config_classes(images_dataset.classes)
    return images_dataset



