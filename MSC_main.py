# -*- coding: utf-8 -*-
"""
@author: zgz

采用的是mask batch的方式，每次模型会吧所有的值都算出来，之后乘以mask，只留下每个batch需要的值；
mask batch的效率应该是比较低的，但是对于gcn好像只有match batch可以用
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch_geometric.data import DataLoader
from torch_geometric import utils
from torch.optim import lr_scheduler
from torch.utils.data import random_split

import argparse
import os
import time
import setproctitle
import sys
import logging
import copy
from functools import reduce

from dataset_processing import *
from gcn_model_global import *
from utils import *
from node2vec import Node2Vec

torch.autograd.set_detect_anomaly(True)


def setup_logging(args):
    # 清空/创建文件
    with open(args.log_file, 'w') as file:
        pass
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def init_model(args):
    # model
    if args.model == 'gcn':
        model = GCN(args).to(args.device)
    elif args.model == 'gcn_h':
        model = GCN_H(args).to(args.device)
    elif args.model == 'gcn_h_cp':
        model = GCN_H_CP(args).to(args.device)
    elif args.model == 'gcn_cp':
        model = GCN_CP(args).to(args.device)
    elif args.model == 'gcn_el':
        model = GCN_EL(args).to(args.device)
    elif args.model == 'gcn_el2':
        model = GCN_EL2(args).to(args.device)
    elif args.model == 'gcn_el_h':
        model = GCN_EL_H(args).to(args.device)
    elif args.model == 'gcn_id':
        model = GCN_ID(args, id_embeddings).to(args.device)
    elif args.model == 'gat':
        model = GAT(args).to(args.device)
    elif args.model == 'sage':
        model = SAGE(args).to(args.device)
    elif args.model == 'gcn_el_cp':
        model = GCN_EL_CP(args).to(args.device)
    elif args.model == 'gcn_el_cp_a':
        model = GCN_EL_CP_A(args).to(args.device)
    elif args.model == 'gcn_el_cp_h':
        model = GCN_EL_CP_H(args).to(args.device)
    elif args.model == 'sagp':
        model = SAGP(args).to(args.device)
    else:
        raise NotImplementedError(args.model)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if not args.finetune or not args.id_embeddings_flag:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.net.parameters()},
            {'params': id_embeddings, 'lr': 0.05 * args.lr}
        ], lr=args.lr, weight_decay=args.weight_decay)
        model.load_state_dict(
            torch.load('../model/model_before_finetune-{}-{}-{}'.format(args.model, args.n_hidden, args.batch_size)))
    if args.scheduler_flag == 1:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=args.T_0,T_mult=args.T_mult,eta_min=args.eta_min)
    elif args.scheduler_flag == 2:
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.5)
    else:
        scheduler = None
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, 
    # verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('Trainable Parameters:', np.sum([p.numel() for p in train_params]))

    return (model, optimizer, scheduler)


def nan_hook(self, inp, output):
    for i, out in enumerate(output):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def train(data, train_loaders, valid_id):
    model.train()
    start = time.time()
    min_loss = 1e5
    patience = 0
    data = data.to(args.device)
    for epoch in range(args.epochs):
        print('Epoch {}:'.format(epoch))
        mae_loss = 0.
        num_iters = len(train_loaders)
        for batch_idx, train_ids in enumerate(train_loaders):
            if args.scheduler_flag:
                scheduler.step(epoch + batch_idx / num_iters)
            optimizer.zero_grad()
            out = model(data)
            loss = F.l1_loss(out[train_ids], data.y[train_ids], reduction='sum')
            mae_loss += F.l1_loss(out[train_ids], data.y[train_ids], reduction='sum').item()/num_train
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
        # mae_loss /= num_train
        print("FOLD {}, Time {:.4f} -- Training loss:{}".format(fold, time_iter, mae_loss))
        val_loss, _, _ = test(model, data, valid_id)
        print("FOLD {}, Time {:.4f} -- Validation loss:{}".format(fold, time_iter, val_loss))
        if val_loss < min_loss:
            if not args.id_embeddings_flag:
                torch.save(model.state_dict(),
                           '../model/model_{}_{}_{}_{}_{}_{}'.format(args.model, args.lr, args.weight_decay,
                                                                     args.n_hidden, args.batch_size, args.n_embedding))
            else:
                if not args.finetune:
                    torch.save(model.state_dict(),
                               '../model/model_before_finetune-{}-{}-{}'.format(args.model, args.n_hidden,
                                                                                args.batch_size))
                else:
                    torch.save(model.state_dict(),
                               '../model/model_after_finetune-{}-{}-{}-{}-{}-{}'.format(args.model, args.lr,
                                                                                        args.weight_decay,
                                                                                        args.n_hidden, args.batch_size,
                                                                                        args.n_embedding))
            print("!!!!!!!!!! Model Saved !!!!!!!!!!")
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break


def test(model, data, test_ids):
    model.eval()
    start = time.time()
    data = data.to(args.device)
    out = model(data)
    mae = F.l1_loss(out[test_ids], data.y[test_ids], reduction='mean')
    mse = F.mse_loss(out[test_ids], data.y[test_ids], reduction='mean')
    rmse = torch.sqrt(mse)
    nrmse = rmse/(torch.max(data.y[test_ids]) - torch.min(data.y[test_ids]))
    return mae.item(), rmse.item(), nrmse.item()


def log_outcome(model, data, test_ids):
    model.eval()
    start = time.time()
    data = data.to(args.device)
    out = model(data)
    mae = F.l1_loss(out[test_ids], data.y[test_ids], reduction='mean')
    mse = F.mse_loss(out[test_ids], data.y[test_ids], reduction='mean')
    rmse = torch.sqrt(mse)
    nrmse = rmse/(torch.max(data.y[test_ids]) - torch.min(data.y[test_ids]))
    return mae.item(), rmse.item(), nrmse.item()


def generate_combination(l1,l2):
    res = []
    for u in l1:
        for v in l2:
            if type(u) is not list:
                u = [u]
            if type(v) is not list:
                v = [v]
            res.append(u+v)
    return res


def generate_grid_search_params(search_params):
    if len(search_params.keys()) == 1:
        return search_params.values()
    else:
        return reduce(generate_combination, search_params.values())


if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Graph convolutional networks for influencer value prediction')
    parser.add_argument('-sd', '--seed', type=int, default=630, help='random seed')
    parser.add_argument('-lr', '--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=125, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('-d', '--dropout_ratio', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-dvs', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str, default='gcn', help='model')
    parser.add_argument('-dp', '--dataset_path', type=str, default='../data/sample2_dataset_norm.npy',
                        help='node feature matrix data path')
    parser.add_argument('-nh', '--n_hidden', type=int, default=32, help='number of hidden nodes in each layer of GCN')
    parser.add_argument('-pr', '--pooling_ratio', type=int, default=0.5, help='Pooling ratio for Pooling layers')
    parser.add_argument('-p', '--patience', type=int, default=150, help='Patience')
    parser.add_argument('-fnf', '--full_node_feature', type=int, default=0,
                        help='whether to include #neighbor as featrue')
    parser.add_argument('--n_id_embedding', type=int, default=5, help='id embedding size')
    parser.add_argument('--n_embedding', type=int, default=20, help='embedding size')
    parser.add_argument('--n_folds', type=int, default=10, help='n_folds')
    parser.add_argument('--id_embeddings_flag', type=int, default=0, help='whether id_embeddings')
    parser.add_argument('--pretrain_flag', type=int, default=0, help='whether pretrain')
    parser.add_argument('--finetune', type=int, default=0, help='whether finetune')
    parser.add_argument('--grid_search', type=int, default=0, help='whether grid_search')
    parser.add_argument('--grid_search_params', type=str, default='', help='grid search params')
    parser.add_argument('--log_file', type=str, default='../log/test.log', help='grid search params')
    parser.add_argument('--lr_decay_steps', type=str, default='[80,180]', help='lr_decay_steps')
    parser.add_argument('--scheduler_flag', type=int, default=0, help='scheduler_flag')
    parser.add_argument('--params_cosinelr', type=str, default='[80,1,0.05]', help='scheduler_flag')
    parser.add_argument('-sr', '--seed_ratio', type=float, default=0.2, help='scheduler_flag')
    parser.add_argument('--write_results_flag', type=int, default=0, help='write_results_flag')
    args = parser.parse_args()

    # 对args做一些处理
    args.lr_decay_steps = eval(args.lr_decay_steps)
    temp = eval(args.params_cosinelr)
    args.T_0, args.T_mult, args.eta_min = temp[0], temp[1], temp[2]
    if args.full_node_feature == 1:
        args.n_demographic = 9
    else:
        args.n_demographic = 8
    args_printer(args)

    # 设定相关信息
    torch.backends.cudnn.deterministic = True  # 每次训练得到相同结果
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cudnn.benchmark = True   # 自动优化卷积实现算法
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    setproctitle.setproctitle('Influencer@zhangguozhen')  # 设定程序名

    logger = setup_logging(args)

    start_time = time.time()

    ############################### Load Data ###############################
    print('------------------------- Loading data -------------------------')
    dataset = create_dataset_global(os.path.join('..', 'data', args.dataset_path), args)
    args.num_features = dataset.num_features
    args.num_edge_features = dataset.num_edge_features
    args.num_communities = int(dataset.community.max().item() + 1)
    args.num_nodes = dataset.x.size(0)
    print(args.num_communities)
    train_ids, val_ids, test_ids = split_train_val_test(args.num_communities, args.n_folds, args.seed)

    if args.grid_search:
        logger.info('Start grid_search')
        search_params = eval(args.grid_search_params)

        best_params = []
        best_mae_folds = []
        best_rmse_folds = []
        best_nrmse_folds = []
        best_mae = 1e5
        for gs_params in generate_grid_search_params(search_params):

            for i,key in enumerate(search_params.keys()):
                exec('args.'+key+'=gs_params[i]')

            mae_folds = []
            rmse_folds = []
            nrmse_folds = []
            for fold in range(args.n_folds):
                train_loaders, num_train = make_batch(train_ids[fold], args.batch_size, args.seed)

                print('\nFOLD {}, train {}, valid {}, test {}'.format(fold, num_train, len(val_ids[fold]), len(test_ids[fold])))

                if args.id_embeddings_flag:
                    print('------------------------- Pre-train id embedding -------------------------')
                    id_embedding_model = Node2Vec(args.num_nodes, args.n_id_embedding, walk_length=8, context_size=3, walks_per_node=10, device='cpu')

                    if args.pretrain_flag:
                        pretrain_id_embedding(id_embedding_model, dataset, 100)
                    else:
                        id_embedding_model.load_state_dict(torch.load('../model/pretrain_model'))

                    id_embeddings = id_embedding_model(torch.arange(args.num_nodes))
                    id_embeddings.detach_()
                    if args.finetune:
                        id_embeddings = torch.nn.Parameter(id_embeddings)
                    print('------------------------- Done! -------------------------')

                print('\n------------------------- Initialize Model -------------------------')
                model, optimizer, scheduler = init_model(args)

                print('\n------------------------- Training -------------------------')
                train(dataset, train_loaders, val_ids[fold])

                print('\n------------------------- Testing -------------------------')
                if not args.id_embeddings_flag:
                    model.load_state_dict(
                        torch.load('../model/model_{}_{}_{}_{}_{}_{}'.format(
                            args.model, args.lr, args.weight_decay,
                            args.n_hidden, args.batch_size, args.n_embedding)))
                else:
                    if not args.finetune:
                        model.load_state_dict(
                            torch.load('../model/model_before_finetune-{}-{}-{}'.format(args.model, args.n_hidden, args.batch_size)))
                    else:
                        model.load_state_dict(torch.load(
                            '../model/model_after_finetune-{}-{}-{}-{}-{}-{}'.format(args.model, args.lr, args.weight_decay,
                                                                                     args.n_hidden, args.batch_size,
                                                                                     args.n_embedding)))
                mae_loss, rmse_loss, nrmse_loss = test(model, dataset, test_ids[fold])
                mae_folds.append(mae_loss)
                rmse_folds.append(rmse_loss)
                nrmse_folds.append(nrmse_loss)

                print('---------------------------------------')
                print('mae_loss: {}'.format(mae_loss))

            args_printer(args)

            logger.info('model:%s, n_hidden:%d, lr:%f, weight_decay:%f, batch_size:%d, n_embedding:%d',
                        args.model, args.n_hidden, args.lr, args.weight_decay, args.batch_size, args.n_embedding)
            logger.info('mae_folds: %s', str(mae_folds))
            # logger.info('rmse_folds: %s', str(rmse_folds))
            # logger.info('nrmse_folds: %s', str(nrmse_folds))
            logger.info('%d-fold cross validation avg mae (+- std): %f (%f)', args.n_folds, np.mean(mae_folds), np.std(mae_folds))
            logger.info('%d-fold cross validation avg rmse (+- std): %f (%f)', args.n_folds, np.mean(rmse_folds), np.std(rmse_folds))
            logger.info('%d-fold cross validation avg nrmse (+- std): %f (%f)', args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds))
            logger.info('---------------------------------------------------------')
            print('Training Finished!')
            print('{}-fold cross validation avg mae (+- std): {} ({})'.format(args.n_folds, np.mean(mae_folds), np.std(mae_folds)))
            print('{}-fold cross validation avg rmse (+- std): {} ({})'.format(args.n_folds, np.mean(rmse_folds), np.std(rmse_folds)))
            print('{}-fold cross validation avg nrmse (+- std): {} ({})'.format(args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds)))
            print('Total train time: {}', time.time()-start_time)
            print('------------------------------------------------------')

            if np.mean(mae_folds) < best_mae:
                best_mae = np.mean(mae_folds)
                best_mae_folds = mae_folds
                best_rmse_folds = rmse_folds
                best_nrmse_folds = nrmse_folds
                best_params = gs_params

        logger.info('Search Done!')
        logger.info('best mae_folds: %s', str(best_mae_folds))
        logger.info('avg mae (+- std): %f (%f)', np.mean(best_mae_folds), np.std(best_mae_folds))
        logger.info('avg rmse (+- std): %f (%f)', np.mean(best_rmse_folds), np.std(best_rmse_folds))
        logger.info('avg nrmse (+- std): %f (%f)', np.mean(best_nrmse_folds), np.std(best_nrmse_folds))
        for i, key in enumerate(search_params.keys()):
            logger.info('Best parameters: %s:%s', str(key), str(best_params[i]))
        print('Search Done!')
        print('Total search time: {}', time.time()-start_time)
        # print('best mae_folds: {}'.format(str(best_mae_folds)))
        print('avg mae (+- std): {} ({})'.format(np.mean(best_mae_folds), np.std(best_mae_folds)))
        print('avg rmse (+- std): {} ({})'.format(np.mean(best_rmse_folds), np.std(best_rmse_folds)))
        print('avg nrmse (+- std): {} ({})'.format(np.mean(best_nrmse_folds), np.std(best_nrmse_folds)))
        for i, key in enumerate(search_params.keys()):
            print('Best parameters: {}:{}'.format(str(key), str(best_params[i])))

    # 不grid_search的情况
    else:
        mae_folds = []
        rmse_folds = []
        nrmse_folds = []
        for fold in range(args.n_folds):
            train_loaders, num_train = make_batch(train_ids[fold], args.batch_size, args.seed)

            print('\nFOLD {}, train {}, valid {}, test {}'.format(fold, num_train, len(val_ids[fold]), len(test_ids[fold])))

            if args.id_embeddings_flag:
                print('------------------------- Pre-train id embedding -------------------------')
                id_embedding_model = Node2Vec(args.num_nodes, args.n_id_embedding, walk_length=8, context_size=3, walks_per_node=10, device='cpu')

                if args.pretrain_flag:
                    pretrain_id_embedding(id_embedding_model, dataset, 100)
                else:
                    id_embedding_model.load_state_dict(torch.load('../model/pretrain_model'))

                id_embeddings = id_embedding_model(torch.arange(args.num_nodes))
                id_embeddings.detach_()
                if args.finetune:
                    id_embeddings = torch.nn.Parameter(id_embeddings)
                print('------------------------- Done! -------------------------')

            print('\n------------------------- Initialize Model -------------------------')
            model, optimizer, scheduler = init_model(args)

            print('\n------------------------- Training -------------------------')
            train(dataset, train_loaders, val_ids[fold])

            print('\n------------------------- Testing -------------------------')
            if not args.id_embeddings_flag:
                model.load_state_dict(
                    torch.load('../model/model_{}_{}_{}_{}_{}_{}'.format(
                        args.model, args.lr, args.weight_decay,
                        args.n_hidden, args.batch_size, args.n_embedding)))
            else:
                if not args.finetune:
                    model.load_state_dict(
                        torch.load(
                            '../model/model_before_finetune-{}-{}-{}'.format(
                                args.model, args.n_hidden, args.batch_size)))
                else:
                    model.load_state_dict(
                        torch.load('../model/model_after_finetune-{}-{}-{}-{}-{}-{}'.format(
                            args.model, args.lr, args.weight_decay,
                            args.n_hidden, args.batch_size,
                            args.n_embedding)))
            mae_loss, rmse_loss, nrmse_loss = test(model, dataset, test_ids[fold])
            mae_folds.append(mae_loss)
            rmse_folds.append(rmse_loss)
            nrmse_folds.append(nrmse_loss)

            print('---------------------------------------')
            print('mae_loss: {}'.format(mae_loss))

        logger.info('model:%s, n_hidden:%d, lr:%f, weight_decay:%f, batch_size:%d, n_embedding:%d',
                    args.model, args.n_hidden, args.lr, args.weight_decay, args.batch_size, args.n_embedding)
        logger.info('mae_folds: %s', str(mae_folds))
        # logger.info('rmse_folds: %s', str(rmse_folds))
        # logger.info('nrmse_folds: %s', str(nrmse_folds))
        logger.info('%d-fold cross validation avg mae (+- std): %f (%f)', args.n_folds, np.mean(mae_folds), np.std(mae_folds))
        logger.info('%d-fold cross validation avg rmse (+- std): %f (%f)', args.n_folds, np.mean(rmse_folds), np.std(rmse_folds))
        logger.info('%d-fold cross validation avg nrmse (+- std): %f (%f)', args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds))
        logger.info('---------------------------------------------------------')

        args_printer(args)
        print('Total train time: {}', time.time()-start_time)
        print('{}-fold cross validation avg mae (+- std): {} ({})'.format(args.n_folds, np.mean(mae_folds), np.std(mae_folds)))
        print('{}-fold cross validation avg rmse (+- std): {} ({})'.format(args.n_folds, np.mean(rmse_folds), np.std(rmse_folds)))
        print('{}-fold cross validation avg nrmse (+- std): {} ({})'.format(args.n_folds, np.mean(nrmse_folds), np.std(nrmse_folds)))
        mae_folds = ['{:.2f}'.format(u) for u in mae_folds]
        mae_folds = [float(u) for u in mae_folds]
        print(mae_folds)
