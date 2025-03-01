import torch, math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor

EPS = float(np.finfo(np.float32).eps)
# 随机数生成器
rng = np.random.default_rng(seed=42)

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:0")

# from torch.nn import MultiheadAttention
# 这就是在每个巻积层进行巻积操作对应论文中的公式（6）但是这个权重就很有意思，论文中没有说这么干
# 这才是实际的巻积过程
import torch

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




def process_(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)

    real = torch.matmul(data, weight)

    data = -1.0 * torch.spmm(mul_L_imag, X_imag)

    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag, X_real)

    imag = torch.matmul(data, weight)

    data = torch.spmm(mul_L_real, X_imag)

    imag += torch.matmul(data, weight)

    return torch.stack([real, imag])

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


class complex_relu_layer_(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer_, self).__init__()

    def complex_relu(self, real, img):
        mask = 1.0 * (real >= 0)
        return mask * real, mask * img

    def forward(self, real, img=None):
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img


class CAGNConv(nn.Module):  # 这个是图巻积层  这就是线性感知机
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag    type list
    """

    def __init__(self, in_c, out_c, K, L_norm_real, L_norm_imag, bias=True, Ritz=None, Qimag=None, Qreal=None,
                 multihopCov=[], num_eig_vec=0, alpha=0.9, adj=[],multihopRes = []):
        super(CAGNConv, self).__init__()

        # L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N  滤波器  各种尺度的laplacian的序列  list
        #  self.Attention = None
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]
        self.num_scale_long = len(multihopCov)
        self.short_diff = multihopRes
        self.short_long = len(self.short_diff)
        self.R = Ritz
        self.Qimag = Qimag
        self.Qreal = Qreal
        self.num_eig_vec = num_eig_vec
        self.multihopCov = multihopCov
        self.out_c = int(out_c / (self.num_scale_long + 1))
        self.weight = nn.Parameter(
            torch.Tensor(K + 1, in_c, self.out_c))
        if self.num_scale_long != 0:
            self.weight_long = nn.Parameter(
                torch.Tensor(self.num_scale_long, in_c, self.out_c))
            1. / math.sqrt(self.weight_long.size(-1))
            nn.init.xavier_uniform_(self.weight_long.data)
            # self.weight_long.data.uniform_(-stdvLong, stdvLong)
        self.weight_res = nn.Parameter(
            torch.Tensor(self.short_long, in_c, out_c))

        nn.init.xavier_uniform_(self.weight_res.data)

        stdv = 1. / math.sqrt(self.weight.size(-1))

        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:

            self.register_parameter("bias", None)

    def _get_spectral_and_res_filters(self):
        T_list = [torch.pow(self.R, i) for i in self.multihopCov + self.short_diff]

        L_normal_real_long = []
        L_normal_imag_long = []
        L_res_real = []
        L_res_imag = []
        with torch.no_grad():
            for i, T in enumerate(T_list):
                # T = T/T.max()

                diag_dd = torch.diag(T)  # diag_dd 是对每个特征值的n次方取对角线元素

                # 计算共用部分以减少矩阵乘法操作
                Q_diag_dd_real = torch.mm(self.Qreal, diag_dd)
                Q_diag_dd_imag = torch.mm(self.Qimag, diag_dd)

                # 计算 real 和 imag 部分，并减少不必要的中间变量
                real_part = torch.mm(Q_diag_dd_real, self.Qreal.T) + torch.mm(Q_diag_dd_imag, self.Qimag.T)
                imag_part = torch.mm(Q_diag_dd_imag, self.Qreal.T) - torch.mm(Q_diag_dd_real, self.Qimag.T)

                # real_part = torch.addmm(torch.mm(self.Qreal, diag_dd), torch.mm(self.Qimag, diag_dd), self.Qreal.T)
                # imag_part = torch.addmm(torch.mm(self.Qimag, diag_dd), -torch.mm(self.Qreal, diag_dd), self.Qimag.T)

                if i < len(self.multihopCov):
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

        future = []
        future_res = []

        for i in range(len(self.mul_L_real)):
            future.append(torch.jit.fork(process,
                                         self.mul_L_real[i], self.mul_L_imag[i],
                                         self.weight[i], X_real, X_imag))

        future_long = []
        if self.multihopCov != [] and self.num_scale_long != 0:
            for i in range(self.num_scale_long):
                future_long.append(torch.jit.fork(process,
                                                  L_normal_real_long[i], L_normal_imag_long[i],
                                                  self.weight_long[i], X_real, X_imag))
        for i in range(self.short_long):
            future_res.append(torch.jit.fork(process,
                                             L_res_real[i], L_res_imag[i],
                                             self.weight_res[i], X_real, X_imag))
        result = []
        result_long = []
        for i in range(len(self.mul_L_real)):  # 我估计是知道为什么用异步去做了，他tensor乘法快不了
            result.append(torch.jit.wait(future[i]))

        if self.multihopCov:
            for i in range(self.num_scale_long):
                result_long.append(torch.jit.wait(future_long[i]))
        result_res = []
        for i in range(self.short_long):
            result_res.append(torch.jit.wait(future_res[i]))

        result = torch.sum(torch.stack(result), dim=0)

        a, b, c = result.size()
        result_long = torch.stack(result_long).view(a, b, -1)

        result = torch.cat([result, result_long], dim=-1)

        result_res = torch.mean(torch.stack(result_res), dim=0)

        real = result[0] + X_real @ torch.cat([self.weight[0], self.weight[1]], dim=-1)

        real = real + result_res[0]

        imag = result[1] + X_imag @ torch.cat([self.weight[0], self.weight[1]], dim=-1)

        imag = imag + result_res[1]

        return real + self.bias, imag + self.bias


class CAGN_Lanczos(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=True, layer=2,
                 num_scale_long=2, Qreal=None, Qimag=None, R=None, num_eig_vec=500, multihopCov=[],
                 dropout=True, adj=[],multihopRes = None):
        """
        :param in_c: int, number of input channels.
        :param num_filter: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(CAGN_Lanczos, self).__init__()

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
        self.multihopCov = multihopCov

        # 这直接创建了第一个巻积层   in_c:原始特征维度数量  out_c 中间隐藏层特征维度数量    L_norm_real：磁化laplacian矩阵的实部
        chebs = [CAGNConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag,
                          multihopCov=self.multihopCov,multihopRes = multihopRes, Ritz=self.R, Qreal=self.Qreal, Qimag=self.Qimag,
                          num_eig_vec=self.num_eig_vec, alpha=1
                          )]

        if activation:            chebs.append(complex_relu_layer_())

        for i in range(1, self.layer):
            chebs.append(
                CAGNConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag,
                         multihopCov=self.multihopCov,multihopRes = multihopRes, Ritz=self.R, Qreal=self.Qreal, Qimag=self.Qimag,
                         num_eig_vec=self.num_eig_vec, alpha=0.5))
            if activation:
                chebs.append(complex_relu_layer_())

        self.Chebs = torch.nn.Sequential(
            *chebs)

        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, real, imag):

        real, imag = self.Chebs((real, imag))
        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute(
            (0, 2, 1))
        x = self.Conv(x)
        x = F.softmax(x, dim=1)
        return x
