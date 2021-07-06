import torch.nn as nn
from utils.functions import ReverseLayerF
import torch

class DAModel(nn.Module):

    def __init__(self, feature_extractor, label_predictor, domain_classifier):
        super(DAModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier


    def forward(self, source_data, target_data, alpha=1):
        # 因为是领域一致的训练其实可以分开来forward, 所以其实Model 并不区分source domain 和target domain
        source_X = self.feature_extractor(source_data)
        target_X = self.feature_extractor(target_data)#6 2 64 64  == #6 512
        SA_class_logits  = self.label_predictor(source_X)

        # 领域分类
        # 我們把source和target混在一起，否則batch_norm会出错
        mixed_data = torch.cat([source_X, target_X], dim=0)  # 按顺序拼接的吗？
        reversed_mixed_data = ReverseLayerF.apply(mixed_data, alpha)

        DA_class_logits = self.domain_classifier(reversed_mixed_data)


        return SA_class_logits, DA_class_logits


    def othersForward(self):
        #用于处理不一样的forward方式

        SA_class_logits, DA_class_logits = (0,0)
        return SA_class_logits, DA_class_logits


    def target_domain_predict(self, test_data):

        target_X = self.feature_extractor(test_data)
        SA_class_logits = self.label_predictor(target_X)

        return SA_class_logits#@jinhui 6 之后要传参
        # 领域分类











