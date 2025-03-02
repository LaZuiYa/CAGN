import pickle
import os
import torch, math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, identity

EPS = float(np.finfo(np.float32).eps)
# 随机数生成器
rng = np.random.default_rng(seed=42)

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:0")


def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    X_real_weighted = torch.matmul(X_real, weight)
    X_imag_weighted = torch.matmul(X_imag, weight)
    # 稀疏矩阵乘法
    L_real_X_real = torch.spmm(mul_L_real.float(), X_real_weighted.float())
    L_imag_X_imag = torch.spmm(mul_L_imag.float(), X_imag_weighted.float())
    L_imag_X_real = torch.spmm(mul_L_imag.float(), X_real_weighted.float())
    L_real_X_imag = torch.spmm(mul_L_real.float(), X_imag_weighted.float())
    # # 计算 real 和 imag
    real = L_real_X_real - L_imag_X_imag
    imag = L_imag_X_real + L_real_X_imag
    # 返回 float16 精度结果（减少显存使用）
    return torch.stack([real, imag])





class complex_relu_layer(nn.Module):  # 这个是复数激活函数层
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img


def matmul_blockwise(A, B, block_size):
    C = torch.zeros(A.size(0), B.size(1), device=A.device)
    for i in range(0, A.size(0), block_size):
        for j in range(0, B.size(1), block_size):
            C[i:i + block_size, j:j + block_size] = torch.mm(A[i:i + block_size], B[:, j:j + block_size])
    return C


def save_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# 你看，他用的是切比雪夫   对应论文公式（5）,这个实际上是滤波器
# 这个cheb巻积就是实部虚部互相乘
class ChebConv(nn.Module):  # 这个是图巻积层  这就是线性感知机
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag    type list
    """

    def __init__(self, in_c, out_c, K, L_norm_real, L_norm_imag, bias=True, Ritz=None, Qimag=None, Qreal=None,
                 long_diff=[], num_eig_vec=0, alpha=0.9, adj=[], edge_index=None):

        # L_norm_real, L_norm_imag = L_norm_real, L_norm_imag
        super(ChebConv, self).__init__()
        # list of K sparsetensors, each is N by N  滤波器  各种尺度的laplacian的序列  list
        #  self.Attention = None
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]
        self.num_scale_long = len(long_diff)
        self.out = int(out_c / (self.num_scale_long + 1))
        self.short_diff = [2, 4]
        self.edge_index = edge_index
        self.short_long = len(self.short_diff)
        self.R = Ritz
        self.Qimag = Qimag
        self.Qreal = Qreal
        self.num_eig_vec = num_eig_vec
        self.long_diff = long_diff

        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.Tensor(K+1 ,in_c, self.out))
        self.adj = adj
        if self.num_scale_long != 0:
            self.weight_long = nn.Parameter(
                torch.Tensor(self.num_scale_long, in_c, self.out))
            stdvLong = 1. / math.sqrt(self.weight_long.size(-1))
            nn.init.xavier_uniform_(self.weight_long.data)

        self.weight_res = nn.Parameter(
            torch.Tensor(self.short_long, in_c, int(out_c / 2)))

        nn.init.xavier_uniform_(self.weight_res.data)

        stdv = 1. / math.sqrt(self.weight.size(-1))

        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:

            self.register_parameter("bias", None)

    def _get_spectral_and_res_filters(self):
        T_list = [torch.pow(self.R, i) for i in self.long_diff + self.short_diff]

        L_normal_real_long = []
        L_normal_imag_long = []
        L_res_real = []
        L_res_imag = []
        with torch.no_grad():
            for i, T in enumerate(T_list):

                diag_dd = torch.diag(T)  # diag_dd 是对每个特征值的n次方取对角线元素

                # 计算共用部分以减少矩阵乘法操作
                Q_diag_dd_real = torch.mm(self.Qreal, diag_dd)
                Q_diag_dd_imag = torch.mm(self.Qimag, diag_dd)

                # 计算 real 和 imag 部分，并减少不必要的中间变量
                real_part = torch.mm(Q_diag_dd_real, self.Qreal.T) + torch.mm(Q_diag_dd_imag, self.Qimag.T)
                imag_part = torch.mm(Q_diag_dd_imag, self.Qreal.T) - torch.mm(Q_diag_dd_real, self.Qimag.T)

                if i < len(self.long_diff):
                    L_normal_real_long.append(real_part)
                    L_normal_imag_long.append(imag_part)
                else:
                    L_res_real.append(real_part)
                    L_res_imag.append(imag_part)
                # 手动释放显存
                del diag_dd, real_part, imag_part
                torch.cuda.empty_cache()

        return L_normal_real_long, L_normal_imag_long, L_res_real, L_res_imag

    def forward(self, data):  # (real, imag)
        """
        :param  inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]
        L_normal_real_long, L_normal_imag_long, L_res_real, L_res_imag = self._get_spectral_and_res_filters()
        for i, j in enumerate(self.long_diff):
            if j % 2 == 0:
                L_normal_real_long[i] = 2 * L_normal_real_long[i] - torch.eye(L_normal_real_long[i].size(0),
                                                                              device='cuda')
                L_normal_imag_long[i] = 2 * L_normal_imag_long[i] - torch.eye(L_normal_imag_long[i].size(0),
                                                                              device='cuda')
        future = []
        future_res = []
        for i in range(len(self.mul_L_real)):
            future.append(torch.jit.fork(process, self.mul_L_real[i], self.mul_L_imag[i],
                                         self.weight[i], X_real, X_imag))

        result = []
        for i in range(len(self.mul_L_real)):  # 我估计是知道为什么用异步去做了，他tensor乘法快不了
            result.append(torch.jit.wait(future[i]))
        future_long = []
        result = torch.sum(torch.stack(result), dim=0)
        for i in range(self.num_scale_long):
            future_long.append(torch.jit.fork(process, L_normal_real_long[i], L_normal_imag_long[i],
                                              self.weight_long[i], X_real, X_real))
        for i in range(self.short_long):
            future_res.append(torch.jit.fork(process,
                                             L_res_real[i], L_res_imag[i],
                                             self.weight_res[i], X_real, X_imag))

        result_cov = []

        if self.long_diff != []:
            for i in range(self.num_scale_long):
                result_cov.append(torch.jit.wait(future_long[i]))
        result_res = []
        for i in range(self.short_long):
            result_res.append(torch.jit.wait(future_res[i]))


        a, b, c = result.size()
        result_cov = torch.stack(result_cov).view(a, b, -1)
        result_res = torch.stack(result_res).view(a, b, -1)

        result = torch.cat([result, result_cov], dim=-1)

        real, imag = result[0] + result_res[0], result[1] + result_res[1]

        real = torch.tanh(real+ self.bias)
        imag = torch.tanh(imag  + self.bias)

        return real, imag



class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=True, layer=2,
                 num_scale_long=2, Qreal=None, Qimag=None, R=None, num_eig_vec=500, long_diff=[10, 20],
                 dropout=True, adj=[], edge_index=None):
        """
        :param in_c: int, number of input channels.  初始节点特征维度大小
        :param num_filter: int, number of hidden channels.   中间隐藏层节点特征嵌入维度
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian    L_norm_real 切比雪夫多项式截断给出的K个滤波器的实部
        """
        super(ChebNet_Edge, self).__init__()

        self.adj = adj
        self.L_norm_imag = L_norm_imag
        self.L_norm_real = L_norm_real
        self.K = K
        self.layer = layer

        self.num_scale_long = num_scale_long
        self.R = R
        self.Qimag = Qimag
        self.Qreal = Qreal
        self.num_eig_vec = num_eig_vec
        self.long_diff = long_diff

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag,
                          long_diff=self.long_diff, Ritz=self.R, Qreal=self.Qreal, Qimag=self.Qimag,
                          num_eig_vec=self.num_eig_vec, alpha=1, edge_index=edge_index
                          )]
        # if activation and (layer != 1):
        #     chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(
                ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag,
                         long_diff=self.long_diff, Ritz=self.R, Qreal=self.Qreal, Qimag=self.Qimag,
                         num_eig_vec=self.num_eig_vec, alpha=0.5, edge_index=edge_index))
            # if activation:
            #     chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.linear = nn.Linear(num_filter * last_dim * 2, label_dim)
        self.dropout = dropout

    def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:, 0]], real[index[:, 1]], imag[index[:, 0]], imag[index[:, 1]]), dim=-1)
        x = torch.tanh(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        x = F.log_softmax(x, dim=1)
        return x


