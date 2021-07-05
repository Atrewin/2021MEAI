
import json
from torch import optim
import argparse, datetime, torch
import traceback
from utils.logger import *

import sys
import os
from os.path import normpath,join,dirname
import numpy as np

from data.data_loder import *
from models.BaseModels import *
from frameworks.AdeversarialTraining import *
from models.DAClassifier import *
def main():
    parser = argparse.ArgumentParser()

    # data url parameters

    parser.add_argument('--source', default='dvd_reviews', help='source file')
    parser.add_argument('--target', default='book_reviews', help='target file')
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
    parser.add_argument('--only_test', action='store_true', help='only test')
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
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = "GCAModel"
    encoder_name = opt.encoder
    max_length = opt.max_length

    # train_log setting
    LOG_PATH = opt.log_root
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    prefix = '-'.join([model_name, encoder_name, opt.source, opt.target])


    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, prefix + "-" + opt.notes + "-" + nowTime + ".txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("log id: {}".format(nowTime))
    logger.info("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    logger.info("model: {}".format(model_name))
    logger.info("encoder: {}".format(encoder_name))
    logger.info("max_length: {}".format(max_length))
    logger.info("#" * 30)
    logger.info("roberta: {}".format(opt.pretrain_ckpt))
    logger.info("Q: {}".format(opt.Q))
    logger.info("source domain: {}".format(opt.source))
    logger.info("target domain: {}".format(opt.target))


    logger.info("start_train_prototypical: {}".format(opt.start_train_prototypical))
    logger.info("start_train_adv: {}".format(opt.start_train_adv))
    logger.info("start_train_dis: {}".format(opt.start_train_dis))

    logger.info("#" * 30)

    #  构造模型   @jinhui 将整个模型框架传入的设计会更加优雅



    ####TODO dataset
    # 这里貌似有异步调用的问题
    source_dataset = get_deep_dataset('data/datasets/train_val/train')
    trarget_dataset = get_deep_dataset('data/datasets/NEU-CLS')
    source_dataloader = DataLoader(source_dataset, batch_size=opt.batch_size, shuffle=True)
    target_dataloader = DataLoader(trarget_dataset, batch_size=opt.batch_size, shuffle=True)

    # 获取训练集信息
    source_classes = 9
    target_classes = 6
    # 准备模型
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor(source_classes).cuda()
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
    farmework = AdeversarialTrainingFramework(model, target_dataloader, target_dataloader, class_criterion, domain_criterion, optimizer, opt)

    farmework.train()
    print("pause")






def getSentenceEncoder(encoder_name, opt):

    if encoder_name == 'cnn':
        try:
            # pre-train embedding
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        max_length = opt.max_length
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length, hidden_size=opt.hidden_size)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        max_length = opt.max_length
        sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt
        max_length = opt.max_length
        filepath, tempfilename = os.path.split(pretrain_ckpt)
        sentence_encoder =RobertaSentenceEncoder(filepath,tempfilename, max_length, cat_entity_rep=opt.cat_entity_rep)
    elif encoder_name == "graph":
        max_length = opt.max_length
        sentence_encoder = GraphSentenceEncoder(max_length, cat_entity_rep=opt.cat_entity_rep,
                                                    mask_entity=opt.mask_entity)

    else:
        raise NotImplementedError

    return sentence_encoder




if __name__ == "__main__":
    main()

    import traceback
    try:
        a = 1
        pass
    except:
        traceback.print_exc()


