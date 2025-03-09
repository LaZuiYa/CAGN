# -*- coding: utf-8 -*-


import argparse
import time
from datetime import datetime

import numpy as np
import torch
# 优化器import torch.optim as optim
from torch import nn

from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS, CoraFull

from layer.Lanczos import *

from layer.CAGN_Layer import CAGN_Lanczos
# internal files
from utils.Citation import *
from utils.preprocess import geometric_dataset_sparse, load_syn
from utils.save_settings import write_log

cuda_device = 0
device = torch.device("cuda:0")
EPS = float(np.finfo(np.float32).eps)

rng = np.random.default_rng(seed=42)


def parse_args():
    parser = argparse.ArgumentParser(description="MagNet Conv (sparse version)")
    parser.add_argument('--log_root', type=str, default='../logs/',
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test',
                        help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/',
                        help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Texas', help='data set selection')
    # WebKB/Cornell++
    # telegram/telegram
    # cora_ml/
    #Cora/cora
    parser.add_argument('--epochs', type=int, default=3000, help='Number of (maximal) training epochs.')
    # (q)ss
    parser.add_argument('--q', type=float, default=0, help='q value for the phase matrix')
    # 方向强度参数是beta吗
    parser.add_argument('--p_q', type=float, default=0.95, help='Direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='Inter-cluster edge probabilities.')
    parser.add_argument('--method_name', type=str, default='Magnet', help='method name')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for training testing split/random graph generation.')
    parser.add_argument('--PreTrain', type=bool, default=False,
                        help='Random seed for training testing split/random graph generation.')
    parser.add_argument('--PreTrain_path', type=str,
                        default="./WebKBWisconsion/06-10-09_53_11/model8.t7",
                        help='Random seed for training testing split/random graph generation.')


    parser.add_argument('--lanczos_step', type=int, default=90, help='K for cheb series')
    parser.add_argument('--multihopCov', type=list, default=[2], help='multihopCov for lanczos')
    parser.add_argument('--multihopRes', type=list, default=[8], help='multihopCov for lanczos')
    parser.add_argument('--MLP', type=str, default="MLP", help='mlp')
    parser.add_argument('--num_filter', type=int, default=512, help='num of filters')

    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=1, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.8, help='dropout prob')

    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')

    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--new_setting', '-NS', action='store_true', help='Whether not to load best settings.')

    parser.add_argument('-activation', '-a', action='store_true', help='if use activation function')
    parser.add_argument('--randomseed', type=int, default=-1, help='if set random seed in training')
    return parser.parse_args()


###################################################################################################################################################################################

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # print(sparse_mx.toarray())
    # torch.from_numpy(ndarray) → Tensor

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


###########################################################################################################################3


def main(args):

    if args.randomseed > 0:

        torch.manual_seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H-%M-%S')

    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) == False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')




    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]

    if load_func == 'WebKB':
        load_func = WebKB
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
    elif load_func == 'WikiCS':
        load_func = WikiCS
    elif load_func == 'cora_ml':
        load_func = citation_datasets
    elif load_func == 'citeseer_npz':
        load_func = citation_datasets

    else:

        load_func = load_syn


    _file_ = args.data_path + args.dataset + '/data' + str(args.q) + '_' + str(args.K) + '_sparse.pk'
    storeLanczos = args.data_path + args.dataset + 'data'

    X, label, train_mask, val_mask, test_mask, L,_ = geometric_dataset_sparse(args.q, args.K,
                                                                                    root=args.data_path + args.dataset,
                                                                                    subset=subset,
                                                                                    dataset=load_func, load_only=False,
                                                                                    save_pk=True)

    _label_ = label - np.amin(label)

    cluster_dim = np.amax(_label_) + 1

    L_img = []

    L_real = []

    L_normal = L[1]

    # adj = L[1]
    # adj.data = np.array(np.ones(len(adj.data)))
    Ritz, Q = restart_lanczos_approxmiate__Tensor(args.lanczos_step, L_normal, datapath=storeLanczos)
    # Ritz = Ritz.astype(np.float32)
    Ritz = torch.tensor(Ritz)
    Q = Q.astype(np.complex128)

    Qimag = torch.FloatTensor(Q.imag).to(device)

    Qreal = torch.FloatTensor(Q.real).to(device)

    Ritz = Ritz.to('cuda')
    for i in range(len(L)):
        # print(np.min(L[1].imag.toarray()*1j))
        L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
        L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))

    label = torch.from_numpy(_label_[np.newaxis]).to(device)
    # label = torch.FloatTensor(label).to(device)
    # print(label.dtype)
    X_img = torch.FloatTensor(X).to(device)

    X_real = torch.FloatTensor(X).to(device)

    criterion = nn.CrossEntropyLoss()
    splits = train_mask.shape[1]

    if len(test_mask.shape) == 1:
        test_mask = np.repeat(test_mask[:, np.newaxis], splits, 1)

    results = np.zeros((splits, 4))

    if args.PreTrain:
        preTrainDict = torch.load(args.PreTrain_path)

    for split in range(splits):
        log_str_full = ''

        # model = ChebNet(X_real.size(-1), L_real, L_img, K = args.K, label_dim=cluster_dim, layer = args.layer,
        #                         activation = True, num_filter = args.num_filter, dropout=args.dropout).to(device)

        model = CAGN_Lanczos(X_real.size(-1), L_real, L_img, K=args.K, label_dim=cluster_dim, layer=args.layer,
                             activation=True, num_filter=args.num_filter, dropout=args.dropout,
                             num_scale_long=len(args.multihopCov), num_eig_vec=args.lanczos_step,
                             multihopCov=args.multihopCov,multihopRes = args.multihopRes ,R=Ritz, Qimag=Qimag, Qreal=Qreal).to(device)
        # model = nn.DataParallel(model)
        if args.PreTrain:
            modelDict = model.state_dict(destination=None)

            modelDict.update(preTrainDict)
            model.load_state_dict(modelDict)


        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_test_acc = 0.0

        train_index = train_mask[:, split]
        val_index = val_mask[:, split]
        test_index = test_mask[:, split]

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0



        for epoch in range(args.epochs):
            start_time = time.time()

            ####################
            # Train
            ####################
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(train_index)

            model.train()
            # model_lanczos.train()
            preds = model(X_real, X_img)
            #            preds[:, :, train_index] = preds[:, :, train_index].type(torch.int32)
            label = label.type(torch.LongTensor).to(device)
            train_loss = criterion(preds[:, :, train_index], label[:, train_index])
            pred_label = preds.max(dim=1)[1]
            train_acc = 1.0 * ((pred_label[:, train_index] == label[:, train_index])).sum().detach().item() / count

            train_loss.backward(retain_graph=True )
            opt.step()
            opt.zero_grad()
            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)
            # scheduler.step()

            ####################
            # Validation
            ####################
            model.eval()
            count, test_loss, test_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(val_index)
            preds = model(X_real, X_img)
            pred_label = preds.max(dim=1)[1]

            test_loss = criterion(preds[:, :, val_index], label[:, val_index])
            test_acc = 1.0 * ((pred_label[:, val_index] == label[:, val_index])).sum().detach().item() / count

            outstrval = ' Test loss:, %.6f, acc:, %.3f,' % (test_loss.detach().item(), test_acc)

            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d ,/, %d ,epoch," % (epoch, args.epochs)) + outstrtrain + outstrval + duration
            log_str_full += log_str + '\n'

            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model' + str(split) + '.t7')
            else:
                early_stopping += 1
            # if early_stopping % 100 == 0:
            # print(save_perform, best_test_err)

            if early_stopping > 500 or epoch == (args.epochs - 1):
                torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
                break

        write_log(vars(args), log_path)

        end_time = time.perf_counter()

        #training_time = end_time - start_time
        #print(f'Training time: {training_time:.2f} seconds')

        ####################
        # Testing
        ####################


        model.load_state_dict(torch.load(log_path + '/model' + str(split) + '.t7'))
        # print(model.state_dict().items())
        model.eval()
        preds = model(X_real, X_img)
        pred_label = preds.max(dim=1)[1]
        np.save(log_path + '/pred' + str(split), pred_label.to('cpu'))

        count = np.sum(val_index)
        acc_train = (1.0 * ((pred_label[:, val_index] == label[:, val_index])).sum().detach().item()) / count

        count = np.sum(test_index)
        acc_test = (1.0 * ((pred_label[:, test_index] == label[:, test_index])).sum().detach().item()) / count


        model.load_state_dict(torch.load(log_path + '/model_latest' + str(split) + '.t7'))
        model.eval()
        preds = model(X_real, X_img)
        pred_label = preds.max(dim=1)[1]
        np.save(log_path + '/pred_latest' + str(split), pred_label.to('cpu'))

        count = np.sum(val_index)
        acc_train_latest = (1.0 * ((pred_label[:, val_index] == label[:, val_index])).sum().detach().item()) / count

        count = np.sum(test_index)
        acc_test_latest = (1.0 * ((pred_label[:, test_index] == label[:, test_index])).sum().detach().item()) / count

        ####################
        # Save testing results
        ####################
        logstr = 'val_acc: ' + str(np.round(acc_train, 3)) + ' test_acc: ' + str(
            np.round(acc_test, 3)) + ' val_acc_latest: ' + str(
            np.round(acc_train_latest, 3)) + ' test_acc_latest: ' + str(np.round(acc_test_latest, 3))
        print(logstr)
        results[split] = [acc_train, acc_test, acc_train_latest, acc_test_latest]
        log_str_full += logstr
        with open(log_path + '/log' + str(split) + '.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":

    args = parse_args()
    if args.debug:
        args.epochs = 1
    if args.dataset[:3] == 'syn':
        if args.dataset[4:7] == 'syn':
            if args.p_q not in [-0.08, -0.05]:
                args.dataset = 'syn/syn' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
            elif args.p_q == -0.08:
                args.p_inter = -args.p_q
                args.dataset = 'syn/syn2Seed' + str(args.seed)
            elif args.p_q == -0.05:
                args.p_inter = -args.p_q
                args.dataset = 'syn/syn3Seed' + str(args.seed)
        elif args.dataset[4:10] == 'cyclic':
            args.dataset = 'syn/cyclic' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
        else:
            args.dataset = 'syn/fill' + str(int(100 * args.p_q)) + 'Seed' + str(args.seed)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays', args.log_path, args.dataset + '/')
    args.log_path = os.path.join(args.log_path, args.method_name, args.dataset)
    if not args.new_setting:
        if args.dataset[:3] == 'syn':
            if args.dataset[4:7] == 'syn':
                setting_dict = pk.load(open('./syn_settings.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 1, 0.9: 4, 0.85: 5, 0.8: 6, 0.75: 7, 0.7: 8, 0.65: 9, 0.6: 10
                }
                if args.p_inter == 0.1:
                    dataset = 'syn/syn' + str(dataset_name_dict[args.p_q])
                elif args.p_inter == 0.08:
                    dataset = 'syn/syn2'
                elif args.p_inter == 0.05:
                    dataset = 'syn/syn3'
                else:
                    raise ValueError('Please input the correct p_q and p_inter values!')
            elif args.dataset[4:10] == 'cyclic':
                setting_dict = pk.load(open('./Cyclic_setting_dict.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 0, 0.9: 1, 0.85: 2, 0.8: 3, 0.75: 4, 0.7: 5, 0.65: 6
                }
                dataset = 'syn/syn_tri_' + str(dataset_name_dict[args.p_q])
            else:
                setting_dict = pk.load(open('./Cyclic_fill_setting_dict.pk', 'rb'))
                dataset_name_dict = {
                    0.95: 0, 0.9: 1, 0.85: 2, 0.8: 3
                }
                dataset = 'syn/syn_tri_' + str(dataset_name_dict[args.p_q]) + '_fill'
            setting_dict_curr = setting_dict[dataset][args.method_name].split(',')
            try:
                args.num_filter = int(setting_dict_curr[setting_dict_curr.index('num_filter') + 1])
            except ValueError:
                pass
            try:
                args.layer = int(setting_dict_curr[setting_dict_curr.index('layer') + 1])
            except ValueError:
                pass
            try:
                args.K = int(setting_dict_curr[setting_dict_curr.index('K') + 1])
            except ValueError:
                pass
            args.lr = float(setting_dict_curr[setting_dict_curr.index('lr') + 1])
            args.q = float(setting_dict_curr[setting_dict_curr.index('q') + 1])
    if os.path.isdir(dir_name) == False:
        try:

            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(
        int(args.num_filter)) + 'q' + str(int(100 * args.q)) + 'layer' + str(int(args.layer))
    args.save_name = save_name


    results = main(args)
    np.save(dir_name + save_name, results)
