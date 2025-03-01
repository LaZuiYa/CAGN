# externel
import torch
import csv, os
import numpy as np
import pickle as pk
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import WebKB, WikipediaNetwork

# internel
from utils.hermitian import *

"""
Node classification is performed in a semi-supervised setting (i.e., access to the test data, but not the test labels, during training). For the datasets Cornell, Texas, Wisconsin, and Telegram we use a 60%/20%/20% training/validation/test split, which might be viewed as more akin to supervised learning, because of the small graph size. For Cora-ML and CiteSeer, we use the same split as [36]. For all of these datasets we use 10 random data splits. For the DSBM datasets, we generated 5 graphs randomly for each type and for each set of parameters, each with 10 different random node splits. We use 20% of the nodes for validation and we vary the proportion of training samples based on the classification difficulty, using 2%, 10%, and 60% of nodes per class for the ordered, cyclic, and noisy cyclic DSBM graphs, respectively, during training, and the rest for testing. Hyperpameters were selected using one of the five generated graphs, and then applied to the other four generated graphs.
"""


def load_cora(q, path='../../dataset/cora/', save_pk=False, K=1):
    # only graph structure without features
    # create the graph, networkx graph
    G = nx.read_edgelist(path + '/cora.edges', nodetype=int, delimiter=',', data=(('weight', float),),
                         create_using=nx.DiGraph())

    # create the label set
    label = {}
    with open(path + '/cora.node_labels') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            label[int(row[0])] = int(row[1])

    # get the adj matrix
    A = nx.adjacency_matrix(G, nodelist=sorted(list(label.keys())), weight='weight')

    L, w, v = hermitian_decomp(A.todense(), q, norm=True)
    multi_order_laplacian = cheb_poly(L, K)

    if save_pk:
        cora = {}
        cora['A'] = A
        cora['L'] = multi_order_laplacian
        cora['eigen_col'] = v
        cora['label'] = label
        # pk.dump(cora, open(path + '/cora'+str(q)+'_'+str(K)+'.pk', 'wb'))

    return A, multi_order_laplacian, v, label


def load_edge_index(file='cora.edges', path='../dataset/cora/'):
    G = nx.read_edgelist(path + file, nodetype=int, delimiter=',', data=(('weight', float),), create_using=nx.DiGraph())
    edge_index = []
    for line in nx.generate_edgelist(G, data=False):
        line = line.split(' ')
        _from_, _to_ = int(line[0]), int(line[1])
        edge_index.append([_from_, _to_])
    edge_index = np.array(edge_index, dtype=np.int).T
    edge_index = torch.from_numpy(edge_index)
    return edge_index


def load_raw_cora(q=0, path="../pygcn/data/cora/", dataset="cora", save_pk=False, K=1, feature_only=False):
    def encode_onehot(labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    if feature_only:
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        return features

    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.toarray()

    L, w, v = hermitian_decomp(adj, q, norm=True)
    multi_order_laplacian = cheb_poly(L, K)

    if save_pk:
        cora = {}
        # cora['A'] = adj.astype('float32')
        cora['L'] = multi_order_laplacian
        cora['eigen_col'] = v
        cora['label'] = labels.astype('uint8')
        # pk.dump(cora, open(path + '/cora_raw'+str(q)+'_'+str(K)+'.pk', 'wb'))

    return adj, multi_order_laplacian, v, labels


def load_syn(root, name=None):
    data = pk.load(open(root + '.pk', 'rb'))
    if os.path.isdir(root) == False:
        try:
            os.makedirs(root)
        except FileExistsError:
            print('Folder exists!')
    return [data]


def geometric_dataset(q, K, root='../dataset/data/tmp/', subset='Cornell', dataset=WebKB,
                      load_only=False, save_pk=True, laplacian=True, gcn_appr=False):
    if subset == '':
        dataset = dataset(root=root)
    else:
        dataset = dataset(root=root, name=subset)

    size = dataset[0].y.size(-1)
    adj = torch.zeros(size, size).data.numpy().astype('uint8')
    adj[dataset[0].edge_index[0], dataset[0].edge_index[1]] = 1

    label = dataset[0].y.data.numpy().astype('int')
    X = dataset[0].x.data.numpy().astype('float32')
    train_mask = dataset[0].train_mask.data.numpy().astype('bool_')
    val_mask = dataset[0].val_mask.data.numpy().astype('bool_')
    test_mask = dataset[0].test_mask.data.numpy().astype('bool_')

    if load_only:
        return X, label, train_mask, val_mask, test_mask

    if isinstance(q, list) == False:
        L, _, _ = hermitian_decomp(adj, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr)
        multi_order_laplacian = cheb_poly(L, K)
    else:
        multi_order_laplacian = []
        for value in q:
            L, _, _ = hermitian_decomp(adj, value, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr)
            multi_l = cheb_poly(L, K)
            multi_order_laplacian.append(multi_l)
        multi_order_laplacian = np.array(multi_order_laplacian).transpose((1, 0, 2, 3))

    save_name = root + '/data' + str(q) + '_' + str(K)
    if laplacian == False:
        save_name += '_P'
    if save_pk:
        data = {}
        data['L'] = multi_order_laplacian
        # pk.dump(data, open(save_name+'.pk', 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return X, label, train_mask, val_mask, test_mask, multi_order_laplacian


# 这用了scite_learn的稀疏矩阵存储方式
def geometric_dataset_sparse(q, K, root='../dataset/data/tmp/', subset='Cornell', dataset=WebKB,
                             load_only=False, save_pk=True, laplacian=True, gcn_appr=False):
    # 选择数据处理方式
    if subset == '':
        dataset = dataset(root=root)
    else:
        # 数据集地址传给数据处理函数
        dataset = dataset(root=root, name=subset)
    # print(dataset[0][0].y)
    dataset = dataset
    b = dataset[0]
    a = b.y
    size = a.size(-1)

    f_node, e_node = dataset[0].edge_index[0], dataset[0].edge_index[1]

    label = dataset[0].y.data.numpy().astype('int')

    X = dataset[0].x.data.numpy().astype('float32')

    train_mask = dataset[0].train_mask.data.numpy().astype('bool_')

    val_mask = dataset[0].val_mask.data.numpy().astype('bool_')

    test_mask = dataset[0].test_mask.data.numpy().astype('bool_')

    if load_only:
        return X, label, train_mask, val_mask, test_mask,dataset[0].edge_index
    # 进行磁化
    try:
        # f_node：edge的源点list e_node：edge的目标点list size：节点数量 q：相位参数 norm:归一化 Ln max_eigen: 最大特征值  edge_weight：是否是带权边
        L = hermitian_decomp_sparse2(f_node, e_node, size, q, norm=True, laplacian=laplacian,
                                    max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=dataset[0].edge_weight)
    except AttributeError:
        L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian,
                                    max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)

    multi_order_laplacian = cheb_poly_sparse(L, K)  # 去作切比雪夫多项式截断   L是磁化的拉普拉斯矩阵   K按照图卷积神经网络的概念实际上是K步的邻域

    save_name = root + '/data' + str(q) + '_' + str(K)  # 保存切比雪夫多项式截断滤波器 实际上可能是因为切比雪夫多项式截断消耗大量的计算资源
    if laplacian == False:
        save_name += '_P'
    if save_pk:
        data = {}
        data['L'] = multi_order_laplacian  # 这就是数据读取的那部
        pk.dump(data, open(save_name + '_sparse.pk', 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return X, label, train_mask, val_mask, test_mask, multi_order_laplacian,dataset[0].edge_index  # 这个就是L


def to_edge_dataset(q, edge_index, K, data_split, size, root='../dataset/data/tmp/', laplacian=True, norm=True,
                    max_eigen=2.0, gcn_appr=False):
    save_name = root + '/edge_' + str(q) + '_' + str(K) + '_' + str(data_split) + '.pk'
    if os.path.isfile(save_name):
        multi_order_laplacian = pk.load(open(save_name, 'rb'))
        return multi_order_laplacian

    adj = torch.zeros(size, size).data.numpy().astype('uint8')
    adj[edge_index[0], edge_index[1]] = 1

    # L, w, v = hermitian_decomp(adj, q, norm=norm, laplacian=laplacian, max_eigen=max_eigen, gcn_appr = gcn_appr)
    # multi_order_laplacian = cheb_poly(L, K)

    # if laplacian == False:
    #    save_name += '_P'

    if isinstance(q, list) == False:
        L, _, _ = hermitian_decomp(adj, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr)
        multi_order_laplacian = cheb_poly(L, K)
    else:
        multi_order_laplacian = []
        for value in q:
            L, _, _ = hermitian_decomp(adj, q, norm=True, laplacian=laplacian, max_eigen=2.0, gcn_appr=gcn_appr)
            multi_l = cheb_poly(L, K)
            multi_order_laplacian.append(multi_l)
        multi_order_laplacian = np.array(multi_order_laplacian).transpose((1, 0, 2, 3))

    # pk.dump(multi_order_laplacian, open(save_name, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return multi_order_laplacian


def to_edge_dataset_sparse(q, edge_index, K, data_split, size, root='../dataset/data/tmp/', laplacian=True, norm=True,
                           max_eigen=2.0, gcn_appr=False):
    save_name = root + '/edge_' + str(q) + '_' + str(K) + '_' + str(data_split) + '333.pk'
    # if os.path.isfile(save_name):
    #     multi_order_laplacian = pk.load(open(save_name, 'rb'))
    #     return multi_order_laplacian

    f_node, e_node = edge_index[0], edge_index[1]
    L = hermitian_decomp_sparse(f_node, e_node, size, q=q, norm=True, laplacian=laplacian, max_eigen=max_eigen,
                                gcn_appr=gcn_appr)
    multi_order_laplacian = cheb_poly_sparse(L, K)
    pk.dump(multi_order_laplacian, open(save_name, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return multi_order_laplacian


def F_in_out(edge_index, size, edge_weight=None):
    if edge_weight is not None:
        a = sp.coo_matrix((edge_weight, edge_index), shape=(size, size)).tocsc()
    else:
        a = sp.coo_matrix((np.ones(len(edge_index[0])), edge_index), shape=(size, size)).tocsc()

    out_degree = np.array(a.sum(axis=0))[0]
    out_degree[out_degree == 0] = 1

    in_degree = np.array(a.sum(axis=1))[:, 0]
    in_degree[in_degree == 0] = 1
    '''
    # can be more efficient
    a = np.zeros((size, size), dtype=np.uint8)
    a[edge_index[0], edge_index[1]] = 1

    out_degree = np.sum(a, axis = 1)
    out_degree[out_degree == 0] = 1
    
    in_degree = np.sum(a, axis = 0)
    in_degree[in_degree == 0] = 1
    '''
    # sparse implementation
    a = sp.csr_matrix(a)
    A_in = sp.csr_matrix(np.zeros((size, size)))
    A_out = sp.csr_matrix(np.zeros((size, size)))
    for k in range(size):
        A_in += np.dot(a[k, :].T, a[k, :]) / out_degree[k]
        A_out += np.dot(a[:, k], a[:, k].T) / in_degree[k]

    A_in = A_in.tocoo()
    A_out = A_out.tocoo()

    edge_in = torch.from_numpy(np.vstack((A_in.row, A_in.col))).long()
    edge_out = torch.from_numpy(np.vstack((A_out.row, A_out.col))).long()

    in_weight = torch.from_numpy(A_in.data).float()
    out_weight = torch.from_numpy(A_out.data).float()
    return to_undirected(edge_index), edge_in, in_weight, edge_out, out_weight


