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


"""
需要明确我的输入是什么：
节点特征矩阵： N x num_feat N表示节点数量， num_feat表示节点特征维度
磁化laplacian矩阵：N x N
节点标签维度 label_dim  这就是最后的分类类别
隐藏层嵌入特征维度  hidden_dim  这是节点特征通过巻积层后的输出维度

"""


def process_Not_sparse(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)

    real = torch.matmul(data, weight)

    data = -1.0 * torch.spmm(mul_L_imag, X_imag)

    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag, X_real)

    imag = torch.matmul(data, weight)

    data = torch.spmm(mul_L_real, X_imag)

    imag += torch.matmul(data, weight)

    return torch.stack([real, imag])


# 巻积过程
def process_(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real.cpu(), X_real.cpu()).cuda()

    real = torch.matmul(data, weight)

    data = -1.0 * torch.spmm(mul_L_imag.cpu(), X_imag.cpu()).cuda()

    real += torch.matmul(data, weight)

    data = torch.spmm(mul_L_imag.cpu(), X_real.cpu()).cuda()

    imag = torch.matmul(data, weight)

    data = torch.spmm(mul_L_real.cpu(), X_imag.cpu()).cuda()

    imag += torch.matmul(data, weight)

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


class ChebConv(nn.Module):  # 这个是图巻积层  这就是线性感知机
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag    type list
    """

    def __init__(self, in_c, out_c, L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N  滤波器  各种尺度的laplacian的序列  list
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]

        self.weight = nn.Parameter(
            torch.Tensor(in_c, out_c))  # [K+1, 1, in_c, out_c]  K+1个滤波器 x num_features  x out_c    这里，因为切比雪夫多项式截断的T0=I

        nn.init.xavier_uniform_(self.weight.data)
        # 注意此处的权重初始化方式
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)

    def forward(self, data):  # (real, imag)
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        future = None

        # [K, B, N, D]  对应与论文公式（6）  结果就会每层会用相同的巻积滤波器    每层的感受野就是K

        future = torch.jit.fork(process_, self.mul_L_real, self.mul_L_imag, self.weight, X_real, X_imag)

        result_real, result_imag = None, None
        # 按照经典的resNet，下面两个应该反过来
        result_real, result_imag = torch.jit.wait(future)

        return result_real + X_real, result_imag + X_imag


class Lanczos_DRC_block(nn.Module):
    def __init__(self, in_c, out_c, Qreal, Qimag,  long_diff,bias=True, Ritz=None,  num_eig_vec=0):
        super(Lanczos_DRC_block, self).__init__()
        """
        in_c : cheb 传过来的特征表示
        out_c: 传给下一个block的特征通道
        """
        self.in_c = in_c  # 输入特征通道
        self.out_c = out_c  # 输出特征通道

        self.Ritz = Ritz  # Ritz pair
        self.Qimag = Qimag  # 归一化 Magnetic Laplacian matirx
        self.Qreal = Qreal
        self.long_diff = long_diff  # 长尺度值
        self.num_eig_vec = num_eig_vec  # 特征值数量

        # 我的目标是分层使用Lanczos_layer+残差想法
        # self.filter_kernel = nn.ModuleList([
        #     nn.Sequential(
        #         *[
        #             nn.Linear(self.num_eig_vec, 1024),
        #             nn.ReLU(),
        #             nn.Linear(1024, 1024),
        #             nn.ReLU(),
        #             nn.Linear(1024, self.num_eig_vec),
        #             nn.ReLU(),
        #         ]
        #     )
        # ])
        #
        # for f in self.filter_kernel:
        #     for ff in f:
        #         if isinstance(ff, nn.Linear):
        #             nn.init.xavier_uniform_(ff.weight.data)
        #             if ff.bias is not None:
        #                 ff.bias.data.zero_()
        # 用于巻积过程的权重
        self.weight = nn.Parameter(
            torch.Tensor(self.in_c, self.out_c)
        )
        nn.init.xavier_uniform_(self.weight.data)

        # if bias:
        #     self.bias=nn.Parameter()
        # else:
        #     self.register_parameter("bias", None)

        self.complex_relu = complex_relu_layer_()

    def _get_spectral_filters(self, ):



        TT = self.Ritz  # 3 x 14 x 14   就是特征值
        TT = torch.pow(TT,self.long_diff).to(device)

        # 将特征值的n次方放到MLP中
        #

        # DD = torch.cat(T_list, dim=1).view(self.num_scale_long, TT.shape[1])
        # DD = self.model_long(DD)

        L_normal_real_long = [
            torch.mm(torch.mm(self.Qreal, torch.diag(TT.squeeze() )), self.Qreal.transpose(0, 1)) + torch.mm(
                torch.mm(self.Qimag, torch.diag(TT.squeeze() )), self.Qimag.transpose(0, 1))]
        L_normal_imag_long = [
            torch.mm(torch.mm(self.Qimag, torch.diag(TT.squeeze() )), self.Qreal.transpose(0, 1)) - torch.mm(
                torch.mm(self.Qreal, torch.diag(TT.squeeze() )), self.Qimag.transpose(0, 1))]

        return L_normal_real_long, L_normal_imag_long

    # 由于nn.Sequnaiul实现了forward函数，所以
    def forward(self, data):
        real, imag = data[0], data[1]

        L_normal_real, L_normal_imag = self._get_spectral_filters()
        L_normal_real = torch.tensor(L_normal_real[0])
        L_normal_imag = torch.tensor(L_normal_imag[0])# 期望是  num_eig_vec X num_eig_vec
        X_real = real  # batch从chebconv传过来，应当是   N X F
        X_imag = imag
        future = None
        future = torch.jit.fork(process_, L_normal_real, L_normal_imag, self.weight, X_real, X_imag)

        result_real, result_imag = None, None
        # 按照经典的resNet，下面两个应该反过来
        result_real, result_imag = torch.jit.wait(future)
        result_real, result_imag = self.complex_relu(result_real, result_imag)

        return real + result_real, imag + result_imag


class ChebDRC_LanczosDRC_Net(nn.Module):  # 这个是主框架
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=True,
                 num_scale_long=2, Qreal=None, Qimag=None, R=None, num_eig_vec=500, long_diff=[10, 20],
                 dropout=True):
        """
        :param in_c: int, number of input channels.  初始节点特征维度大小
        :param num_filter: int, number of hidden channels.   中间隐藏层节点特征嵌入维度
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian    L_norm_real 切比雪夫多项式截断给出的K个滤波器的实部
        """
        super(ChebDRC_LanczosDRC_Net, self).__init__()

        self.L_norm_imag = L_norm_imag
        self.L_norm_real = L_norm_real
        self.K = K
        # self.layer=layer

        self.num_scale_long = num_scale_long
        self.R = R
        self.Qimag = Qimag
        self.Qreal = Qreal
        self.num_eig_vec = num_eig_vec
        self.long_diff = long_diff

        # 这直接创建了第一个巻积层   in_c:原始特征维度数量  out_c 中间隐藏层特征维度数量    L_norm_real：磁化laplacian矩阵的实部
        chebs = [ChebConv(in_c=in_c, out_c=in_c, L_norm_real=L_norm_real[0], L_norm_imag=L_norm_imag[0])]
        # chebs = []

        if activation:
            chebs.append(complex_relu_layer_())

        for i in range(1, K + 1):
            chebs.append(
                ChebConv(in_c=in_c, out_c=in_c, L_norm_real=L_norm_real[i], L_norm_imag=L_norm_imag[i], ))
            if activation:
                chebs.append(complex_relu_layer_())

        self.Chebs = torch.nn.Sequential(
            *chebs)

        lanczos = []
        for i in range(len(self.long_diff)):
            lanczos.append(Lanczos_DRC_block(in_c=in_c, out_c=in_c, Ritz=self.R, Qreal=self.Qreal, Qimag=self.Qimag,
                                             num_eig_vec=num_eig_vec, long_diff=self.long_diff[i]))

        self.resNet_lanczos = torch.nn.Sequential(*lanczos)

        # 全连接层
        last_dim = 2
        self.Conv = nn.Conv1d(in_c * last_dim, label_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, real, imag):

        real, imag = self.Chebs((real, imag))  # feed real , imag to lanczos_DRC_block
        if self.long_diff != []:
            real, imag = self.resNet_lanczos((real, imag))

        x = torch.cat((real, imag), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute(
            (0, 2, 1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x
