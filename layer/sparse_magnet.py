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


def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real,
                      X_real)  # torch.spmm 是 PyTorch 中用于执行稀疏矩阵与密集向量乘积的函数。它接受三个输入参数：稀疏矩阵 sparse, 密集向量 dense, 以及一个可选的 size 参数，用于指定输出张量的形状。
    real = torch.matmul(data, weight)  # torch.matmul 是 PyTorch 框架中用于执行矩阵乘法的函数。它可以接受两个张量作为输入，并返回它们的矩阵乘积。
    data = -1.0 * torch.spmm(mul_L_imag, X_imag)  # i x i=-1
    real += torch.matmul(data, weight)  # 这是把实部和虚部合一起了
    # 实部和虚部互相聚合
    data = torch.spmm(mul_L_imag, X_real)  # 滤波器的虚部和特征的实部聚合
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
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

    def __init__(self, in_c, out_c, K, L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N  滤波器  各种尺度的laplacian的序列  list
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]

        # self.weight 是一个神经网络层中的权重张量，它被初始化为一个形状为 (K + 1, in_c, out_c) 的 PyTorch 张量对象。其中，K 是该层卷积核的大小，in_c 是输入通道数，out_c 是输出通道数。这里使用 nn.Parameter() 方法将张量对象转换为模型参数，并将其赋值给 self.weight。nn.Parameter() 方法会将张量标记为模型的可训练参数，使得在模型的反向传播过程中可以自动计算并更新这些参数的梯度。总之，这一行代码实现了对神经网络层中权重张量的初始化，并将其标记为模型的可训练参数。
        self.weight = nn.Parameter(
            torch.Tensor(K + 1, in_c,
                         out_c))  # [K+1, 1, in_c, out_c]  K+1个滤波器 x num_features  x out_c    这里，因为切比雪夫多项式截断的T0=I

        stdv = 1. / math.sqrt(self.weight.size(-1))  # 最后一维即为out_c
        # 注意此处的权重初始化方式
        self.weight.data.uniform_(-stdv,
                                  stdv)  # self.weight.data.uniform_(-stdv, stdv) 的作用是对神经网络中的权重矩阵进行初始化。其中 self.weight 是一个 PyTorch 张量对象，表示该层的权重矩阵，data 属性表示获取张量对象中存储的数据值，uniform_() 方法则会按照均匀分布随机初始化权重矩阵中的每一个元素。方法调用中的参数 -stdv 和 stdv 表示初始化的范围，具体来说，这个方法会将权重矩阵中的每个元素初始化为一个在区间 [-stdv, stdv] 内均匀分布的随机数。这样的初始化方式可以帮助网络快速收敛，并提高其性能。

        if bias:

            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)



        else:

            self.register_parameter("bias", None)

    def forward(self, data):  # (real, imag)
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []

        for i in range(
                len(self.mul_L_real)):  # 就是K，他是每一层作K个巻积      [K, B, N, D]  对应与论文公式（6）  结果就会每层会用相同的巻积滤波器    每层的感受野就是K

            future.append(torch.jit.fork(process,
                                         # torch.jit.fork 是 PyTorch 框架提供的一个函数，可以用于在 Python 的多线程环境下启动新线程，并在新线程中执行指定的函数。它会将该函数及其输入参数序列化后发送到新线程中执行，而不会阻塞主线程。执行完成后，函数的返回值也会被序列化并发送回主线程。这个函数常常用于在 PyTorch 中实现异步执行模型推理操作，以提高计算性能和效率。例如，在训练深度神经网络时，我们可以使用 fork 启动多个子线程来执行模型推理操作，以避免在主线程中等待每个推理操作完成的时间，从而加速整个训练过程。
                                         self.mul_L_real[i], self.mul_L_imag[i],
                                         self.weight[i], X_real, X_imag))  # 实际上这个权重设置的是对每个滤波器的权重

        result = []

        for i in range(len(self.mul_L_real)):  # 我估计是知道为什么用异步去做了，他tensor乘法快不了
            result.append(torch.jit.wait(future[
                                             i]))  # torch.jit.wait是用于等待一组PyTorch的JIT编译模块完成编译的函数。当我们使用JIT编译模块时，编译过程是异步的，也就是说编译的操作会在后台执行，而不会阻塞当前程序的执行。因此，在某些情况下，我们需要等待所有的JIT编译操作完成后，才能继续执行后续的代码。
        result = torch.sum(torch.stack(result), dim=0)  # 2 X K X N X N

        # 这就是把通过K个滤波器巻积出来的不同的感受野的特征的实数部分和虚数部分分别聚合
        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias


class ChebNet(nn.Module):  # 这个是主框架
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=True, layer=2,
                 dropout=False):
        """
        :param in_c: int, number of input channels.  初始节点特征维度大小
        :param num_filter: int, number of hidden channels.   中间隐藏层节点特征嵌入维度
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian    L_norm_real 切比雪夫多项式截断给出的K个滤波器的实部
        """
        super(ChebNet, self).__init__()

        self.L_norm_imag = L_norm_imag
        self.L_norm_real = L_norm_real
        self.K = K
        self.layer = layer
        # 这直接创建了第一个巻积层   in_c:原始特征维度数量  out_c 中间隐藏层特征维度数量    L_norm_real：磁化laplacian矩阵的实部
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]

        if activation:  # 看来这个就是激活函数
            chebs.append(complex_relu_layer())

        for i in range(1, self.layer):  # 添加巻积层，但是每层的滤波器都是一样的
            chebs.append(
                ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(
            *chebs)  # Sequential((0): ChebConv()(1): ChebConv())  这行代码使用了 PyTorch 中的 nn.Sequential() 方法来构建一个神经网络模型。其中，*chebs 是一个列表参数，表示它包含了多个 Chebyshev 多项式的实例对象。这些 Chebyshev 多项式将按照列表中所给定的顺序串联起来，作为该 Sequential 模型的一系列网络层。最后，该模型的所有网络层按照顺序依次运行，从而完成整个前向传播过程。self.Chebs 是该模型的一个属性，用于保存构建好的 Sequential 模型。
        # 最后一层的维度？

        # 展平层
        last_dim = 2
        self.Conv = nn.Conv1d(num_filter * last_dim, label_dim, kernel_size=1)  # 1维巻积  就是对应与论文中说到的在最后将特征嵌入展平
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))  # 将节点特征的实部和虚部包成元组送进chebConv
        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        # 在PyTorch中，x.unsqueeze(0)是一个张量的方法，用于在指定位置增加一个新的维度。具体来说，该方法会在第0个位置（即最外层）上增加一个新的维度，从而将原始张量变为一个形状为(1, n)（其中n表示原始张量中元素的数量）的二维张量。
        x = x.unsqueeze(0)  # 展平，送进全连接层   x.shape:
        x = x.permute(
            (0, 2, 1))  # 在PyTorch中，x.permute()是一个张量的方法，用于重新排列其维度。参数可以接受一个整数序列来指定新的维度顺序。这个方法不会改变张量的元素值，而只是改变它们的排列方式。
        x = self.Conv(x)  # 展平后送进1维巻积层进行
        x = F.log_softmax(x, dim=1)
        return x


class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2,
                 dropout=False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series8
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()

        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(
                ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2
        self.linear = nn.Linear(num_filter * last_dim * 2, label_dim)
        self.dropout = dropout

    def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:, 0]], real[index[:, 1]], imag[index[:, 0]], imag[index[:, 1]]), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


"""
需要明确我的输入是什么：
节点特征矩阵： N x num_feat N表示节点数量， num_feat表示节点特征维度
磁化laplacian矩阵：N x N
节点标签维度 label_dim  这就是最后的分类类别
隐藏层嵌入特征维度  hidden_dim  这是节点特征通过巻积层后的输出维度

"""


class LanczosNet(nn.Module):
    def __init__(self, in_feat, Laplacian, label_dim=7, hidden_dim=[128, 128], lanczos_step=30,
                 num_layer=2, Tri=None, Qreal=None, Qimag=None,
                 activation=True, dropout=True, diffusion=[1, 2],
                 spectral_filter_kind='MLP'):
        """

        Parameters
        ----------
        in_feat  初始节点输入特征维度
        Laplacian  图磁化laplacian矩阵    type： ndarray
        label_dim  神经网络输出节点分类类别
        hidden_dim  隐藏层节点嵌入维度
        lanczos_step  lanczos算法步长
        activation 激活函数
        long_diffusion
        short_diffusion
        num_layer
        Returns
        -------

        """
        super(LanczosNet, self).__init__()
        # lanczos 迭代步长
        self.lanczos_step = min(Laplacian.shape[0], lanczos_step)
        # 短尺度扩散列表
        self.diffusion = diffusion
        # 长尺度扩散列表
        # 磁化laplacian矩阵
        self.Laplacian = Laplacian
        # 输出分类维度
        self.label_dim = label_dim
        # 隐藏层数量
        self.num_layer = num_layer
        # 是否使用激活函数
        self.activation = activation
        # 是否使用dropout
        self.dropout = dropout
        # 隐藏层节点嵌入维度
        self.hidden_dim = hidden_dim
        # 节点数量
        self.num_node = Laplacian.shape[0]
        # 输入特征维度
        self.input_dim = in_feat

        self.Tri = Tri
        self.Qreal = Qreal
        self.Qimag = Qimag

        self.scal_list = self.diffusion

        # 扩散最大范围
        self.max_diffusion_dist = max(
            self.diffusion) if self.diffusion else None

        # 长程扩散长度
        self.num_scale_diffusion = len(self.diffusion)  # num_scale_long=5
        # 尺度总数
        self.K = self.num_scale_diffusion

        # 滤波器类型
        self.spectral_filter_kind = spectral_filter_kind

        # dim_list[0] 输入特征维度  dim_list[1:7]  中间隐藏层维度   dim_list[:-1] 标签分类维度
        self.dim_list = [self.input_dim] + self.hidden_dim + [self.label_dim]

        self.embedingReal = nn.Embedding(self.Laplacian.shape[0], self.input_dim)
        self.embedingImag = nn.Embedding(self.Laplacian.shape[0], self.input_dim)

        # 这个其实可以换一种玩法，每个尺度一个MLP  这是把T^long送进MLP  T我舍去了虚部
        if self.spectral_filter_kind == 'MLP' and self.num_scale_diffusion > 0:
            self.spectral_filter = nn.ModuleList([
                nn.Sequential(
                    *[
                        # 把一批全送进去卷       原因应该是送进去的是三对角矩阵T  而不是T特征分解后的R(Ritz)   用complexRelu替换普通RELU
                        nn.Linear(self.lanczos_step * self.lanczos_step * self.num_scale_diffusion, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.lanczos_step * self.lanczos_step * self.num_scale_diffusion)
                    ]) for _ in range(self.num_layer)
            ])

        # self.filter = nn.ModuleList([  # 巻积层
        #                                 nn.Linear(dim_list[tt] * (
        #                                         self.num_scale_short + self.num_scale_long ),
        #                                           dim_list[tt + 1]) for tt in range(self.num_layer)
        #                             ] + [nn.Linear(dim_list[-2], dim_list[-1])])  # 最后一层用于作分类
        filter = [LanczosConv(in_feat=self.dim_list[0], out_feat=self.dim_list[1], K=self.K, )]
        if self.activation:
            filter.append(complex_relu_layer())
        for i in range(1, self.num_layer):
            filter.append(LanczosConv(in_feat=self.dim_list[i], out_feat=self.dim_list[i + 1], K=self.K))
            if self.activation:
                filter.append(complex_relu_layer())

        self.conv = nn.Sequential(*filter)

        last_dim = 2
        self.flatt = nn.Conv1d(self.hidden_dim[-1] * last_dim, label_dim, kernel_size=1)

        # attention 注意力层，这个论文里可没说
        # self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

        # 初始化权重
        self._init_param()

    # 初始化权重参数
    def _init_param(self):

        # 注意力层权重
        # for ff in self.att_func:
        #     if isinstance(ff, nn.Linear):
        #         nn.init.xavier_uniform_(ff.weight.data)
        #         if ff.bias is not None:
        #             ff.bias.data.zero_()

        # 滤波器权重
        if self.spectral_filter_kind == 'MLP' and self.num_scale_diffusion > 0:
            for f in self.spectral_filter:
                for ff in f:
                    if isinstance(ff, nn.Linear):
                        nn.init.xavier_uniform_(ff.weight.data)
                        if ff.bias is not None:
                            ff.bias.data.zero_()

    # 生成长程滤波器算子  传进来的是tersor
    def get_spectral_filters(self, T, Qreal, Qimag, layer_idx):
        """ Construct Spectral Filters based on Lanczos Outputs

          Args:
            T: shape  K X K, tridiagonal matrix
            Q: shape  N X K, orthonormal matrix
            layer_idx: int, index of layer

          Returns:
            L: shape  N X N X num_scale  type: list
        """
        L_real = []
        L_imag = []
        T_list_long = []

        # TempT = torch.from_numpy(T).to(device)  # N x N
        TempT = T  # N x N
        # 生成长程
        for ii in range(1, self.max_diffusion_dist + 1):
            if ii in self.diffusion:
                T_list_long += [TempT]
            TempT = torch.mm(TempT, TempT)

        # 算fi(T^long)
        if self.spectral_filter_kind == 'MLP':  # torch.cat(T_list,dim=1)  K x (num_scale_long)
            DD = self.spectral_filter[layer_idx](torch.cat(T_list_long, dim=1).view(-1))  # DD  K  x K x num_scale_long
            DD = DD.view(T.shape[0], T.shape[1], self.num_scale_diffusion)

            # construct symmetric output  DD + DDT
            DD = (DD + DD.transpose(0, 1)) * 0.5

            for ii in range(self.num_scale_diffusion):
                real = torch.mm(Qreal, DD[:, :, ii]).mm(Qreal.transpose(0, 1)) + torch.mm(Qimag, DD[:, :, ii]).mm(
                    Qimag.transpose(0, 1))
                imag = torch.mm(Qimag, DD[:, :, ii]).mm(Qreal.transpose(0, 1)) - torch.mm(Qreal, DD[:, :, ii]).mm(
                    Qimag.transpose(0, 1))
                # 注意此时L是一个列表，列表元素是ndarray

                L_real += [real]
                L_imag += [imag]
        else:

            TT = torch.cat(T_list_long, dim=1).view(T.shape[0], T.shape[1], self.num_scale_diffusion)
            for ii in range(self.num_scale_diffusion):
                real = torch.mm(Qreal, TT[:, :, ii]).mm(Qreal.transpose(0, 1)) + torch.mm(Qimag, TT[:, :, ii]).mm(
                    Qimag.transpose(0, 1))
                imag = torch.mm(Qimag, TT[:, :, ii]).mm(Qreal.transpose(0, 1)) - torch.mm(Qreal, TT[:, :, ii]).mm(
                    Qimag.transpose(0, 1))

                L_real += [real]
                L_imag += [imag]

        torch.cuda.empty_cache()
        return torch.stack(L_real, dim=0), torch.stack(L_imag, dim=0)  # L_real 和 L_imag  num_scale_long x N x N

    # 正向传播开始
    def forward(self, real, imag):
        msg = []
        for tt in range(self.num_layer):
            if (tt % 2 == 0):

                if self.num_scale_diffusion > 0:
                    L_real, L_imag = self.get_spectral_filters(self.Tri, self.Qreal, self.Qimag,
                                                               tt / 2)  # (num_scale_short+num_scale_long) x N x N

                    # 短尺度滤波操作是直接用磁化laplacian矩阵和特征乘

                    real, imag = self.conv[tt]((real, imag), L_real, L_imag)
                else:
                    real, imag = self.conv[tt](real, imag)

        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.flatt(x)
        x = F.log_softmax(x, dim=1)
        return x


class LanczosConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """

    def __init__(self, in_feat, out_feat, K, bias=True):
        super(LanczosConv, self).__init__()
        # filter = [LanczosConv(in_feat=dim_list[0], out_feat=dim_list[1], K=self.K, )]

        # self.weight 是一个神经网络层中的权重张量，它被初始化为一个形状为 (K + 1, in_c, out_c) 的 PyTorch 张量对象。其中，K 是该层卷积核的大小，in_c 是输入通道数，out_c 是输出通道数。这里使用 nn.Parameter() 方法将张量对象转换为模型参数，并将其赋值给 self.weight。nn.Parameter() 方法会将张量标记为模型的可训练参数，使得在模型的反向传播过程中可以自动计算并更新这些参数的梯度。总之，这一行代码实现了对神经网络层中权重张量的初始化，并将其标记为模型的可训练参数。
        self.weight = nn.Parameter(
            torch.Tensor(K, in_feat, out_feat))  # [K+1, 1, in_c, out_c]  K+1个滤波器 x num_features  x out_c

        stdv = 1. / math.sqrt(self.weight.size(-1))  # 最后一维即为out_c

        self.weight.data.uniform_(-stdv,
                                  stdv)  # self.weight.data.uniform_(-stdv, stdv) 的作用是对神经网络中的权重矩阵进行初始化。其中 self.weight 是一个 PyTorch 张量对象，表示该层的权重矩阵，data 属性表示获取张量对象中存储的数据值，uniform_() 方法则会按照均匀分布随机初始化权重矩阵中的每一个元素。方法调用中的参数 -stdv 和 stdv 表示初始化的范围，具体来说，这个方法会将权重矩阵中的每个元素初始化为一个在区间 [-stdv, stdv] 内均匀分布的随机数。这样的初始化方式可以帮助网络快速收敛，并提高其性能。

        if bias:

            self.bias = nn.Parameter(torch.Tensor(1, out_feat))
            nn.init.zeros_(self.bias)



        else:

            self.register_parameter("bias", None)

    def forward(self, data, L_norm_real, L_norm_imag):  # (real, imag)
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N  滤波器
        self.mul_L_real = L_norm_real  # [K, N, N]
        self.mul_L_imag = L_norm_imag  # [K, N, N]

        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []

        for i in range(len(self.mul_L_real)):  # [K, B, N, D]  对应与论文公式（6）  结果就会每层会用相同的巻积滤波器    每层的感受野就是K

            future.append(torch.jit.fork(process,
                                         # torch.jit.fork 是 PyTorch 框架提供的一个函数，可以用于在 Python 的多线程环境下启动新线程，并在新线程中执行指定的函数。它会将该函数及其输入参数序列化后发送到新线程中执行，而不会阻塞主线程。执行完成后，函数的返回值也会被序列化并发送回主线程。这个函数常常用于在 PyTorch 中实现异步执行模型推理操作，以提高计算性能和效率。例如，在训练深度神经网络时，我们可以使用 fork 启动多个子线程来执行模型推理操作，以避免在主线程中等待每个推理操作完成的时间，从而加速整个训练过程。
                                         self.mul_L_real[i], self.mul_L_imag[i],
                                         self.weight[i], X_real, X_imag))  # 实际上这个权重设置的是对每个滤波器的权重

        result = []

        for i in range(len(self.mul_L_real)):  # 我估计是知道为什么用异步去做了，他tensor乘法快不了
            result.append(torch.jit.wait(future[
                                             i]))  # torch.jit.wait是用于等待一组PyTorch的JIT编译模块完成编译的函数。当我们使用JIT编译模块时，编译过程是异步的，也就是说编译的操作会在后台执行，而不会阻塞当前程序的执行。因此，在某些情况下，我们需要等待所有的JIT编译操作完成后，才能继续执行后续的代码。
        result = torch.sum(torch.stack(result), dim=0)

        # 这就是把通过K个滤波器巻积出来的不同的感受野的特征的实数部分和虚数部分分别聚合
        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias
