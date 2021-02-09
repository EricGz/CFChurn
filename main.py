# -*- coding: utf-8 -*-
"""
@author: zgz

"""

import time
import setproctitle
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from utils import create_dataset, split_train_val_test, make_batch, args_printer
from models import CFChurn

import mlflow
from mlflow.tracking import MlflowClient

import warnings

warnings.filterwarnings("ignore")


def init_model(args):
    '''
    Define and initialize model, optimizer, and scheduler
    '''
    if args.model == 'cf_churn':
        model = CFChurn(args).to(args.device)
    else:
        raise NotImplementedError(args.model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_params = list(filter(lambda x: x.requires_grad, model.parameters()))
    print('#Trainable Parameters:', np.sum([p.numel() for p in train_params]))
    mlflow.log_param('param_num', sum(p.numel() for p in model.parameters()))

    return (model, optimizer)


def loss_func5(ids, pred_y, y, pred_y_cf, y_cf, id_cf, pred_T, t, pred_y0, pred_y1):
    return ((1 + args.imbalance_penalty_t1 * y[ids, :]) * t[ids, :] *
            (F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
             F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :] * args.cf_weight1) +

            (1 + args.imbalance_penalty_t0 * y[ids, :]) * (1 - t[ids, :]) *
            (F.binary_cross_entropy(pred_y[ids, :], y[ids, :], reduction='none') +
             F.binary_cross_entropy(pred_y_cf[ids, :], y_cf[ids, :], reduction='none') * id_cf[ids, :] * args.cf_weight2) +

            F.binary_cross_entropy(pred_T[ids, :], t[ids, :], reduction='none') * args.treatment_weight +

            torch.max(torch.cat([pred_y0[ids, :] - pred_y1[ids, :], torch.zeros(pred_y0[ids, :].size(0), 1).to(args.device)], dim=-1), dim=-1).values.view(-1, 1) * args.causal_weight * id_cf[ids, :])


def test(model, data, test_ids):
    model.eval()
    data = data.to(args.device)
    if args.model in {'cf_churn'}:
        pred_y, pred_y_cf, pred_y0, pred_y1, pred_T, h_ci, h_si = model(data)
        loss = loss_func5(test_ids, pred_y, data.y, pred_y_cf, data.y_cf, data.id_cf, pred_T, data.t, pred_y0, pred_y1)
    else:
        raise NotImplementedError("{} have not been implemented!".format(args.model))
    loss = torch.mean(loss).item()

    prediction = pred_y.round()
    TP = ((prediction[test_ids] == 1) *
          (data.y[test_ids, :] == 1)).sum().item()
    TN = ((prediction[test_ids] == 0) *
          (data.y[test_ids, :] == 0)).sum().item()
    FN = ((prediction[test_ids] == 0) *
          (data.y[test_ids, :] == 1)).sum().item()
    FP = ((prediction[test_ids] == 1) *
          (data.y[test_ids, :] == 0)).sum().item()

    precision = 0
    recall = 0
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    auc = roc_auc_score(data.y[test_ids, :].cpu().detach().numpy(),
                        prediction[test_ids].cpu().detach().numpy())
    return loss, precision, recall, accuracy, auc


def train(data, train_loaders, valid_id):
    model.train()
    start = time.time()
    min_loss = 1e5
    patience = 0
    n_step = 0
    data = data.to(args.device)
    for epoch in range(args.epochs):
        # model.train()
        loss_log = 0.
        # num_iters = len(train_loaders)
        train_ids_all = []
        n_train = 0
        for batch_idx, train_ids in enumerate(train_loaders):
            # if args.scheduler_flag:
            #     scheduler.step(epoch + batch_idx / num_iters)
            train_ids_all += list(train_ids)
            n_train += len(train_ids)
            optimizer.zero_grad()
            if args.model in {'cf_churn'}:
                pred_y, pred_y_cf, pred_y0, pred_y1, pred_T, h_ci, h_si = model(data)
                loss = loss_func5(train_ids, pred_y, data.y, pred_y_cf, data.y_cf, data.id_cf, pred_T, data.t, pred_y0, pred_y1)
            else:
                raise NotImplementedError("{} have not been implemented!".format(args.model))

            loss = torch.sum(loss)
            loss_log += loss.item()
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start

        loss_log = loss_log / n_train
        _, precision, recall, accuracy, auc = test(model, data, train_ids_all)
        if epoch % 10 == 0:
            print('Epoch {}:'.format(epoch))
            print("FOLD {}, Time {:.4f} -- Training loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                fold, time_iter, loss_log, precision, recall, accuracy, auc))
        mlflow.log_metric(key='epoch', value=epoch, step=n_step)
        mlflow.log_metric(key='train_loss', value=loss_log, step=n_step)
        mlflow.log_metric(key='train_accuracy', value=accuracy, step=n_step)
        mlflow.log_metric(key='train_precision', value=precision, step=n_step)
        mlflow.log_metric(key='train_recall', value=recall, step=n_step)
        mlflow.log_metric(key='train_AUC', value=auc, step=n_step)

        val_loss, precision, recall, accuracy, auc = test(
            model, data, valid_id)
        if epoch % 10 == 0:
            print("FOLD {}, Time {:.4f} -- Validation loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                fold, time_iter, val_loss, precision, recall, accuracy, auc))
        mlflow.log_metric(key='val_loss', value=val_loss, step=n_step)
        mlflow.log_metric(key='val_accuracy', value=accuracy, step=n_step)
        mlflow.log_metric(key='val_precision', value=precision, step=n_step)
        mlflow.log_metric(key='val_recall', value=recall, step=n_step)
        mlflow.log_metric(key='val_AUC', value=auc, step=n_step)

        if val_loss < min_loss:
            if epoch % 10 == 0:
                print("!!!!!!!!!! Model Saved !!!!!!!!!!")
            torch.save(model.state_dict(), '../model/model_{}_{}'.format(
                args.exp_name, args.model_name_suffix))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break
        n_step += 1


if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(
        description='Graph convolutional networks for influencer value prediction')
    parser.add_argument('--exp_name', type=str, default='test', help='exp_name')
    parser.add_argument('-dp', '--dataset_path', type=str, default='sample1_dataset_norm.npy',
                        help='node feature matrix data path')
    parser.add_argument('-dir', '--data_directory', type=str, default='/data',
                        help='data_directory')
    parser.add_argument('-sd', '--seed', type=int,
                        default=666, help='random seed')
    parser.add_argument('-lr', '--lr', type=float,
                        default=0.002, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=125, help='batch size')
    parser.add_argument('-wd', '--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('-e', '--epochs', type=int,
                        default=400, help='number of epochs')
    parser.add_argument('-d', '--dropout', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('-dvs', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str,
                        default='base', help='model')
    parser.add_argument('--n_folds', type=int, default=5, help='n_folds')
    parser.add_argument('-ne', '--n_embedding', type=int,
                        default=10, help='embedding size')
    parser.add_argument('-ncc', '--n_channels_c', type=int,
                        default=4, help='n_channels_c')
    parser.add_argument('-nh', '--n_hidden', type=int, default=32,
                        help='number of hidden nodes in each layer of GCN')
    parser.add_argument('--heads', type=int, default=1, help='heads')
    parser.add_argument('-p', '--patience', type=int,
                        default=150, help='Patience')
    parser.add_argument('-ip', '--imbalance_penalty', type=float,
                        default=1, help='Data imbalance penalty')
    parser.add_argument('-ip0', '--imbalance_penalty_t0', type=float,
                        default=1, help='Data imbalance penalty')
    parser.add_argument('-ip1', '--imbalance_penalty_t1', type=float,
                        default=1, help='Data imbalance penalty')
    parser.add_argument('-cw', '--causal_weight', type=float, default=0)
    parser.add_argument('-cw1', '--causal_weight1', type=float, default=0)
    parser.add_argument('-cw2', '--causal_weight2', type=float, default=0)
    parser.add_argument('-cfw1', '--cf_weight1', type=float, default=0)
    parser.add_argument('-cfw2', '--cf_weight2', type=float, default=0)
    parser.add_argument('-dw', '--disentanglement_weight', type=float, default=0)
    parser.add_argument('-gw', '--gae_weight', type=float, default=0)
    parser.add_argument('-tw', '--treatment_weight', type=float, default=0)
    parser.add_argument('-bn', '--batch_norm', type=int, default=0)
    parser.add_argument('-lh', '--l_hidden', type=int, default=5)
    parser.add_argument('-act', '--activation', type=str, default='leaky_relu')
    args = parser.parse_args()

    args_printer(args)

    # 设定相关信息
    torch.backends.cudnn.deterministic = True  # 每次训练得到相同结果
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed_all(args.seed)   # 为所有GPU设置随机种子
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False   # 自动优化卷积实现算法
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    setproctitle.setproctitle('Churn@zhangguozhen')  # 设定程序名

    start_time = time.time()

    print('------------------------- Loading data -------------------------')
    dataset = create_dataset('../data/' + args.dataset_path)
    args.n_discrete_features = int(dataset.discrete_x.size(1))
    args.n_continous_features = int(dataset.continous_x.size(1))
    args.n_edge_features = int(dataset.edge_attr.size(1))
    args.num_targets = int(dataset.y.size(0))
    args.num_nodes = dataset.discrete_x.size(0)
    args.model_name_suffix = ''.join(random.sample(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e'], 8))

    train_ids, val_ids, test_ids = split_train_val_test(
        args.num_targets, args.n_folds, args.seed)

    mlflow.set_tracking_uri('../exp_logs/mlflow')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(args.exp_name)
    except:
        experiments = client.get_experiment_by_name(args.exp_name)
        EXP_ID = experiments.experiment_id

    with mlflow.start_run(experiment_id=EXP_ID):
        mlflow.log_params(vars(args))

        test_loss, test_precision, test_recall, test_accuracy, test_auc = [], [], [], [], []
        for fold in range(args.n_folds):
            train_loaders, num_train = make_batch(
                train_ids[fold], args.batch_size, args.seed)

            print('\nFOLD {}, train {}, valid {}, test {}'.format(
                fold, num_train, len(val_ids[fold]), len(test_ids[fold])))

            print('\n------------- Initialize Model -------------')
            model, optimizer = init_model(args)

            print('\n------------- Training -------------')
            train(dataset, train_loaders, val_ids[fold])

            print('\n------------- Testing -------------')
            model.load_state_dict(
                torch.load('../model/model_{}_{}'.format(
                    args.exp_name, args.model_name_suffix)))
            loss, precision, recall, accuracy, auc = test(
                model, dataset, test_ids[fold])
            test_loss.append(loss)
            test_precision.append(precision)
            test_recall.append(recall)
            test_accuracy.append(accuracy)
            test_auc.append(auc)

            print('---------------------------------------')
            print("Test loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}".format(
                loss, precision, recall, accuracy, auc))

        args_printer(args)
        print('Total train time: {}', time.time() - start_time)
        print('{}-fold cross validation avg loss:{}, precision:{}, recall:{}, accuracy:{}, AUC:{}'.format(
            args.n_folds, np.mean(test_loss), np.mean(test_precision),
            np.mean(test_recall), np.mean(test_accuracy), np.mean(test_auc)))

        test_accuracy_trunc = [int(u*10000)/10000 for u in test_accuracy]
        test_auc_trunc = [int(u*10000)/10000 for u in test_auc]
        mlflow.log_param(key="test_auc_all", value=str(test_auc_trunc))
        mlflow.log_param(key="test_acc_all", value=str(test_accuracy_trunc))
        mlflow.log_metric(key='test_loss', value=np.mean(test_loss))
        mlflow.log_metric(key='test_accuracy', value=np.mean(test_accuracy))
        mlflow.log_metric(key='test_precision', value=np.mean(test_precision))
        mlflow.log_metric(key='test_recall', value=np.mean(test_recall))
        mlflow.log_metric(key='test_AUC', value=np.mean(test_auc))
