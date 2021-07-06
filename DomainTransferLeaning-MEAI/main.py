from torch.utils.data import DataLoader
import torch
import torch.utils.data as data
import json
from torch import optim
import argparse, datetime, torch
import traceback
from utils.logger import *

from data.data_loder import *
from models.BaseModels import *
from frameworks.AdeversarialTraining import *
from models.DAClassifier import *
def main():
    parser = argparse.ArgumentParser()

    # data url parameters

    parser.add_argument('--source', default='data/datasets/train_val/train', help='source file')
    parser.add_argument('--target', default='data/datasets/NEU-CLS', help='target file')
    # 几点
    # model training parameters
    parser.add_argument('--trainN', default=2, type=int, help='N in train')  # 固定trainN=2
    parser.add_argument('--N', default=2, type=int, help='N way')  # 固定N=2
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int, help='Num of query per class')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size')
    parser.add_argument('--train_iter', default=20000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=10, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=100, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=1, type=int, help='val after training how many iters')
    parser.add_argument('--encoder', default='bert', help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=200, type=int, help='max length')  # 数据集的平均长度
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw', help='sgd / adam / adamw')  # 改 sgd
    parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--fp16', action='store_true', help='use nvidia apex fp16')
    parser.add_argument('--only_test_target', action='store_true', help='only test')
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true',
                        help='do not use dropout after BERT (still has dropout in BERT).')

    # experiment
    parser.add_argument('--mask_entity', action='store_true', help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')
    parser.add_argument('--adv', default=1, type=int, help='if adversarial training')
    parser.add_argument('--alpha', default=None, type=int, help='weight of adv loss training')

    # 切分时间设计
    parser.add_argument('--start_train_prototypical', default=-1, type=int, help='iter to start train prototypical')
    parser.add_argument('--start_train_adv', default=500000, type=int, help="iter to start add adv")
    parser.add_argument('--start_train_dis', default=-1, type=int, help='iter to start train discriminator.')
    parser.add_argument('--is_old_graph_feature', default=0, type=int, help='1 if old graph feature ')
    parser.add_argument('--ignore_graph_feature', default=0, type=int, help='1 if ignore graph feature ')
    parser.add_argument('--ignore_bert_feature', default=0, type=int, help='1 if ignore bert feature ')

    # log模块设计
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    parser.add_argument("--notes", type=str, default='', help='Root directory for all logging.')

    opt = parser.parse_args()

    # train_log setting
    LOG_PATH = opt.log_root
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    prefix = ""



    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, prefix + "-" + opt.notes + "-" + nowTime + ".txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("log id: {}".format(nowTime))
    logger.info("source domain: {}".format(opt.source))
    logger.info("target domain: {}".format(opt.target))
    logger.info("#" * 30)

    #  构造模型   @jinhui 将整个模型框架传入的设计会更加优雅



    ####TODO dataset
    # 这里貌似有异步调用的问题
    source_dataset = get_deep_dataset(opt.source)

    trarget_dataset = get_deep_dataset(opt.target)
    source_dataloader = DataLoader(source_dataset, batch_size=opt.batch_size, shuffle=True)
    target_dataloader = DataLoader(trarget_dataset, batch_size=opt.batch_size, shuffle=True)

    # 获取训练集信息
    source_classes = len(source_dataset.classes)
    target_classes = len(trarget_dataset.classes)

    class_nums = max(source_classes, target_classes)
    # 准备模型
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor(class_nums).cuda()
    domain_classifier = DomainClassifier().cuda()


    # 封装models
    model = DAModel(feature_extractor, label_predictor, domain_classifier)
    # 准备评估函数
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # 准备优化器
    # optimizer_F = optim.Adam(feature_extractor.parameters())
    # optimizer_C = optim.Adam(label_predictor.parameters())
    # optimizer_D = optim.Adam(domain_classifier.parameters())
    optimizer = optim.Adam(model.parameters())


    # 将训练对象封装如farmework
    farmework = AdeversarialTrainingFramework(model, source_dataloader, target_dataloader,
                                              class_criterion, domain_criterion, optimizer,
                                              source_classes, target_classes, opt)
    if(not opt.only_test_target):
        farmework.train()
        acc = farmework.target_inference(model)
        print(acc)
    else:
        acc = farmework.target_inference(model, opt.load_ckpt)
        logger.info("result on " + opt.target + " is {:6.4f}".format(acc))
        # print(acc)








if __name__ == "__main__":
    main()

    import traceback
    try:
        a = 1
        pass
    except:
        traceback.print_exc()


