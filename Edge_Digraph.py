import numpy as np
import pickle as pk
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
from torch_geometric.utils import to_undirected

# internal files
from layer.DiGCN import *
from utils.edge_data import generate_dataset_2class, generate_dataset_3class, in_out_degree, get_appr_directed_adj, get_second_directed_adj, link_prediction_evaluation
from utils.preprocess import geometric_dataset, load_edge_index, load_syn
from utils.save_settings import write_log
# select cuda device if available
from utils.Citation import load_citation_link

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="link prediction baseline--Digraph(NeurIPS2020)")

    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='cora_ml/', help='data set selection')
    
    
    parser.add_argument('--split_prob', type=lambda s: [float(item) for item in s.split(',')], default="0.15,0.05", help='random drop for testing/validation/training edges (for 3-class classification only)')
    parser.add_argument('--task', type=int, default=2, help='2: 2-class classification 3: 3-class classification')

    parser.add_argument('--method_name', type=str, default='DiG', help='method name')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--num_class_link', type=int, default=2,
                        help='number of classes for link direction prediction(2 or 3).')

    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--epochs', type=int, default=1500, help='training epochs')
    parser.add_argument('--num_filter', type=int, default=512, help='num of filters')
    #parser.add_argument('-to_undirected', '-tud', action='store_true', help='if convert graph to undirecteds')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha teleport prob')
    #parser.add_argument('-dgrees', '-d', action='store_true', help='if use in degree+outdegree as feature')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')    
    return parser.parse_args()

def acc(pred, label):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def main(args): 

    date_time = datetime.now().strftime('%m-%d-%H-%M-%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)

    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    if load_func == 'WebKB':
        load_func = WebKB
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikipediaNetwork':
        load_func = WikipediaNetwork
        dataset = load_func(root=args.data_path, name=subset)
    elif load_func == 'WikiCS':
        load_func = WikiCS
        dataset = load_func(root=args.data_path)
    elif load_func == 'cora_ml':
        dataset = load_citation_link(root='../dataset/data/tmp/cora_ml/cora_ml.npz')
    elif load_func == 'citeseer':
        dataset = load_citation_link(root='../dataset/data/tmp/citeseer_npz/citeseer_npz.npz')
        #load telegram/synthetic here
    else:
        dataset = load_syn(args.data_path + args.dataset, None)

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)
        
    # load dataset
    if 'dataset' in locals():
        data = dataset[0]
        edge_index = data.edge_index

    size = torch.max(edge_index).item()+1
    # generate edge index dataset
    #if args.task == 2:
    #    datasets = generate_dataset_2class(edge_index, splits = 10, test_prob = args.drop_prob)
    #else:
    save_file = args.data_path + args.dataset + '/' + subset
    datasets = generate_dataset_3class(edge_index, size, save_file, splits = 10, probs = args.split_prob, task=args.task, label_dim=args.num_class_link)

    if args.task != 2:
        results = np.zeros((10, 4))
    else:
        results = np.zeros((10, 4, 5))
    for i in range(10):
        log_str_full = ''
        edges = datasets[i]['graph']
        
        ########################################
        # initialize model and load dataset
        ########################################
        x = in_out_degree(edges, size).to(device)
        
        # get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None)
        edge_index1, edge_weights1 = get_appr_directed_adj(args.alpha, edges.long(), size, x.dtype)
        edge_index1 = edge_index1.to(device)
        edge_weights1 = edge_weights1.to(device)
        if args.method_name[-2:] == 'ib':
            edge_index2, edge_weights2 = get_second_directed_adj(edges.long(), size, x.dtype)
            edge_index2 = edge_index2.to(device)
            edge_weights2 = edge_weights2.to(device)
            edges = (edge_index1, edge_index2)
            edge_weight = (edge_weights1, edge_weights2)
            del edge_index2, edge_weights2
        else:
            edges = edge_index1
            edge_weight = edge_weights1
        del edge_index1, edge_weights1
        
        ########################################
        # initialize model and load dataset
        ########################################
        if not args.method_name[-2:] == 'ib':
            model = DiGCNet(x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)
        else:
            model = DiGCNet_IB(x.size(-1), args.num_class_link, hidden=args.num_filter).to(device)

        #model = nn.DataParallel(graphmodel)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        y_train = datasets[i]['train']['label']
        y_val   = datasets[i]['validate']['label']
        y_test  = datasets[i]['test']['label']
        y_train = torch.from_numpy(y_train).long().to(device)
        y_val   = torch.from_numpy(y_val).long().to(device)
        y_test  = torch.from_numpy(y_test).long().to(device)

        train_index = torch.from_numpy(datasets[i]['train']['pairs']).to(device)
        val_index = torch.from_numpy(datasets[i]['validate']['pairs']).to(device)
        test_index = torch.from_numpy(datasets[i]['test']['pairs']).to(device)
        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            if early_stopping > 500:
                break
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.train()
            out = model(x, edges, train_index, edge_weight)

            train_loss = F.nll_loss(out, y_train)
            pred_label = out.max(dim = 1)[1]            
            train_acc  = acc(pred_label, y_train)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            outstrtrain = 'Train loss: %.6f, acc: %.3f' % (train_loss.detach().item(), train_acc)
            
            ####################
            # Validation
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.eval()
            out = model(x, edges, val_index, edge_weight)

            test_loss  = F.nll_loss(out, y_val)
            pred_label = out.max(dim = 1)[1]            
            test_acc   = acc(pred_label, y_val)

            outstrval = ' Test loss: %.6f, acc: %.3f' % (test_loss.detach().item(), test_acc)            
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = ("%d / %d epoch" % (epoch, args.epochs))+outstrtrain+outstrval+duration
            #print(log_str)
            log_str_full += log_str + '\n'
            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model'+str(i)+'.t7')
            else:
                early_stopping += 1

        write_log(vars(args), log_path)
        torch.save(model.state_dict(), log_path + '/model_latest'+str(i)+'.t7')
        if args.task != 2:
            ####################
            # Testing
            ####################
            model.load_state_dict(torch.load(log_path + '/model'+str(i)+'.t7'))
            model.eval()
            out = model(x, edges, val_index, edge_weight)[:,:2]
            pred_label = out.max(dim = 1)[1]
            val_acc = acc(pred_label, y_val)
    
            out = model(x, edges, test_index, edge_weight)[:,:2]
            pred_label = out.max(dim = 1)[1]
            test_acc = acc(pred_label, y_test)
        
            model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
            model.eval()
            out = model(x, edges, val_index, edge_weight)[:,:2]
            pred_label = out.max(dim = 1)[1]
            val_acc_latest = acc(pred_label, y_val)
        
            out = model(x, edges, test_index, edge_weight)[:,:2]
            pred_label = out.max(dim = 1)[1]
            test_acc_latest = acc(pred_label, y_test)
            ####################
            # Save testing results
            ####################
            log_str = ('val_acc: {val_acc:.4f}, '+'test_acc: {test_acc:.4f}, ')
            log_str1 = log_str.format(val_acc = val_acc, test_acc = test_acc)
            log_str_full += log_str1

            log_str = ('val_acc_latest: {val_acc_latest:.4f}, ' + 'test_acc_latest: {test_acc_latest:.4f}, ' )
            log_str2 = log_str.format(val_acc_latest = val_acc_latest, test_acc_latest = test_acc_latest)
            log_str_full += log_str2 + '\n'
            print(log_str1+log_str2)

            results[i] = [val_acc, test_acc, val_acc_latest, test_acc_latest]
        else:
            model.load_state_dict(torch.load(log_path + '/model'+str(i)+'.t7'))
            model.eval()
            out_val = model(x, edges, val_index, edge_weight)
            out_test = model(x, edges, test_index, edge_weight)
            [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
                [test_acc_full, test_acc, test_auc, 
                test_f1_micro, test_f1_macro]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
            
            model.load_state_dict(torch.load(log_path + '/model_latest'+str(i)+'.t7'))
            model.eval()
            out_val = model(x, edges, val_index, edge_weight)
            out_test = model(x, edges, test_index, edge_weight)
            [[val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest, val_f1_macro_latest],
                            [test_acc_full_latest, test_acc_latest, test_auc_latest, 
                            test_f1_micro_latest, test_f1_macro_latest]] = link_prediction_evaluation(out_val, out_test, y_val, y_test)
            ####################
            # Save testing results
            ####################
            log_str = ('val_acc_full:{val_acc_full:.4f}, val_acc: {val_acc:.4f}, Val_auc: {val_auc:.4f},'
                       + 'val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, '
                       + 'test_acc_full:{test_acc_full:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}, '
                       + 'test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}')
            log_str = log_str.format(val_acc_full=val_acc_full,
                                     val_acc=val_acc, val_auc=val_auc, val_f1_micro=val_f1_micro,
                                     val_f1_macro=val_f1_macro, test_acc_full=test_acc_full,
                                     test_acc=test_acc, test_auc=test_auc,
                                     test_f1_micro=test_f1_micro, test_f1_macro=test_f1_macro)
            log_str_full += log_str + '\n'
            print(log_str)
            # data = data+ log_str_full
            log_str = (
                    'val_acc_full_latest:{val_acc_full_latest:.4f}, val_acc_latest: {val_acc_latest:.4f}, Val_auc_latest: {val_auc_latest:.4f},'
                    + 'val_f1_micro_latest: {val_f1_micro_latest:.4f}, val_f1_macro_latest: {val_f1_macro_latest:.4f},'
                    + 'test_acc_full_latest:{test_acc_full_latest:.4f}, test_acc_latest: {test_acc_latest:.4f}, test_auc_latest: {test_auc_latest:.4f}, '
                    + 'test_f1_micro_latest: {test_f1_micro_latest:.4f}, test_f1_macro_latest: {test_f1_macro_latest:.4f}')
            log_str = log_str.format(val_acc_full_latest=val_acc_full_latest,
                                     val_acc_latest=val_acc_latest, val_auc_latest=val_auc_latest,
                                     val_f1_micro_latest=val_f1_micro_latest, val_f1_macro_latest=val_f1_macro_latest,
                                     test_acc_full_latest=test_acc_full_latest,
                                     test_acc_latest=test_acc_latest, test_auc_latest=test_auc_latest,
                                     test_f1_micro_latest=test_f1_micro_latest,
                                     test_f1_macro_latest=test_f1_macro_latest)
            log_str_full += log_str + '\n'
            print(log_str)

            results[i] = [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
                          [test_acc_full, test_acc, test_auc, test_f1_micro, test_f1_macro],
                          [val_acc_full_latest, val_acc_latest, val_auc_latest, val_f1_micro_latest,
                           val_f1_macro_latest],
                          [test_acc_full_latest, test_acc_latest, test_auc_latest, test_f1_micro_latest,
                           test_f1_macro_latest]]
        
        with open(log_path + '/log'+str(i)+'.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1
    
    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'alpha' + str(int(100*args.alpha))+ 'task' + str(int(args.task)) + 'link' + str(int(args.num_class_link))
    args.save_name = save_name

    args.log_path = os.path.join(args.log_path,args.method_name, args.dataset)
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')

    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')

    results = main(args)
    np.save(dir_name+save_name, results)