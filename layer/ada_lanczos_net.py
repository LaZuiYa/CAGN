import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.data_helper import check_dist

EPS = float(np.finfo(
    np.float32).eps)  # 这行代码使用了NumPy库来计算32位浮点数的机器精度（machine epsilon），也就是表示为二进制时，能够与1之间的最小差值。np.finfo(np.float32).eps返回一个非负浮点数，表示32位浮点数的机器精度。
__all__ = ['AdaLanczosNet']


class AdaLanczosNet(nn.Module):

    def __init__(self, config):
        super(AdaLanczosNet, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim  # 64
        self.hidden_dim = config.model.hidden_dim  # [128, 128, 128, 128, 128, 128, 128]
        self.output_dim = config.model.output_dim  # 输入64维的节点原始特征，通过7个图巻积层最后映射成16维的节点特征嵌入
        self.num_layer = config.model.num_layer  #
        self.num_atom = config.dataset.num_atom  # 实际上构建adjMatrix的方法很好，节点数量就取包含节点最大的分子的节点数量
        self.num_edgetype = config.dataset.num_bond_type  # 边的数量也同理
        self.dropout = config.model.dropout if hasattr(config.model,
                                                       'dropout') else 0.0
        self.short_diffusion_dist = check_dist(config.model.short_diffusion_dist)  # 检查扩散长度的有效性
        self.long_diffusion_dist = check_dist(config.model.long_diffusion_dist)
        self.max_short_diffusion_dist = max(
            self.short_diffusion_dist) if self.short_diffusion_dist else None
        self.max_long_diffusion_dist = max(
            self.long_diffusion_dist) if self.long_diffusion_dist else None
        self.num_scale_short = len(self.short_diffusion_dist)  # num_scale_short=3
        self.num_scale_long = len(self.long_diffusion_dist)  # num_scale_long=5
        self.num_eig_vec = config.model.num_eig_vec  # K=20
        self.spectral_filter_kind = config.model.spectral_filter_kind  # MLP对准确度的贡献应该比较大
        self.use_reorthogonalization = config.model.use_reorthogonalization if hasattr(
            config, 'use_reorthogonalization') else True
        self.use_power_iteration_cap = config.model.use_power_iteration_cap if hasattr(
            config, 'use_power_iteration_cap') else True

        self.input_dim = self.num_atom  # zhe

        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([  # 巻积层
                                        nn.Linear(dim_list[tt] * (
                                                self.num_scale_short + self.num_scale_long + self.num_edgetype + 1),
                                                  dim_list[tt + 1]) for tt in range(self.num_layer)
                                    ] + [nn.Linear(dim_list[-2], dim_list[-1])])  # 最后一层用于作分类

        self.embedding = nn.Embedding(self.num_atom, self.input_dim)  # 70 x 70

        # spectral filters
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            # N.B.: one can modify the filter size based on GPU memory consumption
            self.spectral_filter = nn.ModuleList([
                nn.Sequential(*[
                    # 这里有问题 为什么是 num_eig_vec x num_eig_vec x num_scale_long     原因应该是送进去的是三对角矩阵T  而不是T特征分解后的R(Ritz)
                    nn.Linear(self.num_eig_vec * self.num_eig_vec * self.num_scale_long, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, self.num_eig_vec * self.num_eig_vec * self.num_scale_long)
                ]) for _ in range(self.num_layer)
            ])

        # attention 注意力层，这个论文里可没说
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            for f in self.spectral_filter:
                for ff in f:
                    if isinstance(ff, nn.Linear):
                        nn.init.xavier_uniform_(ff.weight.data)
                        if ff.bias is not None:
                            ff.bias.data.zero_()

    # 对应论文公式9   或者Graph Kernel这一部分
    def _get_graph_laplacian(self, node_feat, adj_mask):
        """ Compute graph Laplacian

          Args:
            node_feat: float tensor, shape B X N X D    3 x 14 x 70
            adj_mask: float tensor, shape B X N X N, binary mask, should contain self-loop  3 x 14 x 14

          Returns:
            L: float tensor, shape B X N X N
        """
        batch_size = node_feat.shape[0]  # 3
        num_node = node_feat.shape[1]  # 14
        dim_feat = node_feat.shape[2]  # 70

        # compute pairwise distance
        idx_row, idx_col = np.meshgrid(range(num_node), range(
            num_node))  # np.meshgrid()是一个Numpy的函数，它用于生成坐标矩阵。具体来说，如果给出两个一维数组x和y，则np.meshgrid(x, y)将返回两个二维数组X和Y，其中X[i, j] = x[j]，而Y[i, j] = y[i]，对于所有的0 <= i < len(y)和0 <= j < len(x)。
        idx_row, idx_col = torch.Tensor(idx_row.reshape(-1)).long().to(node_feat.device), torch.Tensor(
            idx_col.reshape(-1)).long().to(
            node_feat.device)  # 最终，这行代码会生成一个与 idx_row 具有相同数据的 PyTorch 张量，并将其类型设置为 64 位整数类型。然后，该张量将被移动到 node_feat 所在的设备上。
        # 这是 Graph Kernel
        diff = node_feat[:, idx_row, :] - node_feat[:, idx_col, :]  # shape B X N^2 X D    3 x 196 x 70
        dist2 = (diff * diff).sum(dim=2)  # shape B X N^2

        # sigma2, _ = torch.median(dist2, dim=1, keepdim=True) # median is sometimes 0
        # sigma2 = sigma2 + 1.0e-7

        sigma2 = torch.mean(dist2, dim=1, keepdim=True)  # 这是求一行的均值  3 x 1

        A = torch.exp(-dist2 / sigma2)  # shape B X N^2  3 x 196    dist2的每一个元素除以均值
        A = A.reshape(batch_size, num_node, num_node) * adj_mask  # shape B X N X N   adj_mask  3 x 14 x 14
        row_sum = torch.sum(A, dim=2, keepdim=True)  # 3 x14 x 1
        pad_row_sum = torch.zeros_like(row_sum)  # 3 x 14 x 1
        pad_row_sum[row_sum == 0.0] = 1.0
        alpha = 0.5
        D = 1.0 / (row_sum + pad_row_sum).pow(alpha)  # shape B X N X 1
        L = D * A * D.transpose(1, 2)  # shape B X N X N  3 x 14 x 14

        return L  # 返回 laplacian矩阵

    def _lanczos_layer(self, A, mask=None):
        """ Lanczos layer for symmetric matrix A

          Args:
            A: float tensor, shape B X N X N
            mask: float tensor, shape B X N
            掩码的价值是什么？

          Returns:
            T: float tensor, shape B X K X K
            Q: float tensor, shape B X N X K
        """
        batch_size = A.shape[0]  # 3
        num_node = A.shape[1]
        lanczos_iter = min(num_node, self.num_eig_vec)  # 你lanczos不能超过矩阵大小，按照论上，可以无穷大

        # initialization
        alpha = [None] * (lanczos_iter + 1)  # 这段代码创建了一个名为"alpha"的列表，其长度为"lanczos_iter + 1"。其中每个列表项都被初始化为"None"。
        beta = [None] * (lanczos_iter + 1)
        Q = [None] * (lanczos_iter + 2)

        beta[0] = torch.zeros(batch_size, 1, 1).to(A.device)  # beta[i] 3 x 1 x 1
        Q[0] = torch.zeros(batch_size, num_node, 1).to(A.device)  # Q(0): B X N X 1              Q[i] 3 x 14 x 1
        Q[1] = torch.randn(batch_size, num_node, 1).to(A.device)  # 随机生成 q1

        if mask is not None:  # mask 3 x 14
            mask = mask.unsqueeze(dim=2).float()
            Q[1] = Q[1] * mask
            # 归一化
        Q[1] = Q[1] / torch.norm(Q[1], 2, dim=1,
                                 keepdim=True)  # Returns the matrix norm or vector norm of a given tensor.

        w = None
        C = None
        Qtemp = None
        roundErrorMax = None
        roundErrorNorm = None

        # Lanczos loop
        lb = 1.0e-4  # 验证收敛性的
        valid_mask = []  # 1 x 3 x 1 x 1
        for ii in range(1, lanczos_iter + 1):
            z = torch.bmm(A, Q[ii])  # shape B X N X 1  3 x 14 x 1   z = Sqj
            alpha[ii] = torch.sum(Q[ii] * z, dim=1, keepdim=True)  # shape B X 1 X 1   γj = qjT * z
            z = z - alpha[ii] * Q[ii] - beta[ii - 1] * Q[ii - 1]  # shape B X N X 1

            # EPS = float(np.finfo(np.float32).eps)
            if (ii > 1):
                Qtemp = torch.cat(Q[1:ii], dim=2)
                C = torch.bmm(Qtemp.transpose(1, 2), Qtemp).cpu().data.numpy()
                # print(np.max(np.abs(C-np.eye(C.shape[2])[np.newaxis,:,:])))

                # κi =max 1≤j≤i−1 |ˆ qH iˆ qj |.  ki<EPS^0.5
                w = np.max(np.abs(C - np.eye(C.shape[2])[np.newaxis, :, :])) > EPS ** 0.5
                # print(np.max(np.abs(C-np.eye(C.shape[2])[np.newaxis,:,:]))>EPS**0.5)
                # print(np.diagonal(np.abs(C),axis1=1,axis2=2))
                # print((C-torch.eye(C.shape[1])[None,:,:].to(A.device)))

                # print(torch.abs(C.sub(torch.eye(C.shape[1])[None,:,:].to(A.device))))

            #  EPS=1.1920928955078125e-07
            # I-QT*Q  去掉对角线上的QjT * QjT
            # roundErrorMax=torch.abs(C.sub(torch.eye(C.shape[1])[None,:,:].to(A.device))).cpu().data.numpy()
            # print(np.where(torch.abs(C).cpu().data.numpy()==1)  )

            if (self.use_reorthogonalization and ii > 1) and w:
                # N.B.: Gram Schmidt does not bring significant difference of performance
                # 这种gram-schmdit正交化不是全正交化
                def _gram_schmidt(xx, tt):  # z = z − ∑j−1 i=1 zTqiqi
                    # xx shape B X N X 1
                    for jj in range(1, tt):  # 注意作者使用了经典的Gram-Schmdit算法，这个算法在代码中的形式和modify-Gram-Schmdit算法形式非常像
                        xx = xx - torch.sum(
                            xx * Q[jj], dim=1, keepdim=True) / (torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * \
                             Q[jj]  # 这EPS是舍入误差了
                    return xx

                def modify_gram_schmdit(uu, tt):
                    # xx B x N x 1  传进来的是z
                    # 对q1~qi重正交化
                    for ii in range(2, tt):
                        for jj in range(1, ii):
                            Q[ii].sub(torch.sum(Q[ii] * Q[jj], dim=1, keepdim=True) / (
                                        torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * Q[jj])
                        Q[ii].div((torch.sum(Q[ii] * Q[ii], dim=1, keepdim=True) + EPS))

                    # uu=xx.clone()
                    for jj in range(1, tt):  # proj=(v,u)/(u,u)
                        uu.sub(torch.sum(uu * Q[jj], dim=1, keepdim=True) / (
                                    torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * Q[jj])
                    return uu

                # do Gram Schmidt process twice   这个作两次有什么意义吗？除非是说明经典gram-shmidt算法有问题

                z = modify_gram_schmdit(z, ii)

            beta[ii] = torch.norm(z, p=2, dim=1, keepdim=True)  # shape B X 1 X 1    βj = ‖z‖2

            # N.B.: once lanczos fails at ii-th iteration, all following iterations
            # are doomed to fail
            tmp_valid_mask = (beta[ii] >= lb).float()  # shape
            if ii == 1:
                valid_mask += [tmp_valid_mask]
            else:
                valid_mask += [valid_mask[-1] * tmp_valid_mask]

            # early stop
            Q[ii + 1] = (z * valid_mask[-1]) / (beta[ii] + EPS)

        # get alpha & beta
        alpha = torch.cat(alpha[1:], dim=1).squeeze(dim=2)  # shape B X T
        beta = torch.cat(beta[1:-1], dim=1).squeeze(dim=2)  # shape B X (T-1)

        valid_mask = torch.cat(valid_mask, dim=1).squeeze(dim=2)  # shape B X T   3 x 14
        idx_mask = torch.sum(valid_mask, dim=1).long()  # 14 x 5 x 2
        if mask is not None:
            idx_mask = torch.min(idx_mask, torch.sum(mask, dim=1).squeeze().long())

        for ii in range(batch_size):
            if idx_mask[ii] < valid_mask.shape[1]:
                valid_mask[ii, idx_mask[ii]:] = 0.0

        # remove spurious columns
        alpha = alpha * valid_mask
        beta = beta * valid_mask[:, :-1]
        T = []
        for ii in range(batch_size):
            T += [
                torch.diag(alpha[ii]) + torch.diag(beta[ii], diagonal=1) + torch.diag(
                    beta[ii], diagonal=-1)
            ]

        T = torch.stack(T, dim=0)  # shape B X T X T
        Q = torch.cat(Q[1:-1], dim=2)  # shape B X N X T
        Q_mask = valid_mask.unsqueeze(dim=1).repeat(1, Q.shape[1], 1)

        # remove spurious rows   为了统一图的大小
        for ii in range(batch_size):
            if idx_mask[ii] < Q_mask.shape[1]:
                Q_mask[ii, idx_mask[ii]:, :] = 0.0

        Q = Q * Q_mask

        # pad 0 when necessary
        if lanczos_iter < self.num_eig_vec:
            pad = (0, self.num_eig_vec - lanczos_iter, 0,
                   self.num_eig_vec - lanczos_iter)
            T = F.pad(T, pad)
            pad = (0, self.num_eig_vec - lanczos_iter)
            Q = F.pad(Q, pad)

        return T, Q

    # 这个是用来算 fi(T^n)的  就是把三对角矩阵送进MLP
    def _get_spectral_filters(self, T, Q, layer_idx):
        """ Construct Spectral Filters based on Lanczos Outputs

          Args:
            T: shape B X K X K, tridiagonal matrix
            Q: shape B X N X K, orthonormal matrix
            layer_idx: int, index of layer

          Returns:
            L: shape B X N X N X num_scale
        """
        # multi-scale diffusion
        L = []
        T_list = []
        TT = T  # 3 x 14 x 14

        for ii in range(1, self.max_long_diffusion_dist + 1):
            if ii in self.long_diffusion_dist:
                T_list += [TT]

            TT = torch.bmm(TT, T)  # shape B X K X K
        # T_list=[T^L1,T^L2,.....T^Lmax]     5 x 3 x  20 x 20
        # spectral filter
        if self.spectral_filter_kind == 'MLP':

            DD = self.spectral_filter[layer_idx](torch.cat(T_list, dim=2).view(T.shape[0], -1))  # 3 x 2000
            DD = DD.view(T.shape[0], T.shape[1], T.shape[2],
                         self.num_scale_long)  # shape: B X K X K X C  3 x 20 x 20 x 5

            # construct symmetric output  DD + DDT
            DD = (DD + DD.transpose(1, 2)) * 0.5

            for ii in range(self.num_scale_long):
                L += [Q.bmm(DD[:, :, :, ii]).bmm(Q.transpose(1, 2))]  # Q * fi(T^L|long_scale|) * Q^H    5 x 3 x 14 x 14
        else:
            for ii in range(self.num_scale_long):
                L += [Q.bmm(T_list[ii]).bmm(Q.transpose(1, 2))]

        return torch.stack(L, dim=3)

    # 前向传播训练
    def forward(self, node_feat, L, label=None, mask=None):
        """
          shape parameters:
            batch size = B
            embedding dim = D
            max number of nodes within one mini batch = N
            number of edge types = E
            number of predicted properties = P

          Args:
            node_feat: long tensor, shape B X N
            L: float tensor, shape B X N X N X (E + 1)   3 x 14 x 14 x 7
            label: float tensor, shape B X P    3 x 16
            mask: float tensor, shape B X N
        """
        batch_size = node_feat.shape[0]  # 3
        num_node = node_feat.shape[1]  # 14
        state = self.embedding(node_feat)  # shape: B X N X D  3 x 14 x 70
        # L 3 x 14 x 14 x 7
        if self.num_scale_long > 0:
            # compute graph Laplacian for simple graph
            adj = torch.zeros_like(L[:, :, :, 0])  # get L of the simple graph
            adj[L[:, :, :, 0] != 0.0] = 1.0  # 所谓sample graph就是没有边属性的邻接表，若节点之间有连接就设为1
            Le = self._get_graph_laplacian(state, adj)  # Le = D^(-1/2) x A x D^(-1/2)

            # Lanczos Iteration
            T, Q = self._lanczos_layer(Le, mask)
            # T 3 x 20 x 20  Q 3 x 14 x 20
        ###########################################################################
        # Graph Convolution
        ###########################################################################
        # propagation
        for tt in range(self.num_layer):
            msg = []

            if self.num_scale_long > 0:  # 生成长程滤波器，这就是把三对角矩阵T和Q送进MLP
                Lf = self._get_spectral_filters(T, Q, tt)

            # short diffusion  短尺度直接拿laplacian矩阵去乘
            if self.num_scale_short > 0:
                tmp_state = state
                for ii in range(1, self.max_short_diffusion_dist + 1):
                    tmp_state = torch.bmm(L[:, :, :, 0], tmp_state)
                    if ii in self.short_diffusion_dist:
                        msg += [tmp_state]

            # long diffusion
            if self.num_scale_long > 0:
                for ii in range(self.num_scale_long):
                    msg += [torch.bmm(Lf[:, :, :, ii], state)]  # shape: B X N X D

            # edge type
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, ii], state)]  # shape: B X N X D     3 x 14 x 70
            # 初始msg 15 x 3 x 14 x 70   15=(num_scale_short + num_scale_long + num_edge_type+1)     msg 15 x 3 x 14 x 128
            # msg += [Q]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size,
                                             -1)  # 第一层：42 x 1050  第二层：42 x 1920          42 = 3x14   1050=15x70    1920=128 x 15
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)  # 3 x 14 x 128
            state = F.dropout(state, self.dropout, training=self.training)

        # output
        state = state.view(batch_size * num_node, -1)  # 42 x 128 BN=42 D=128
        y = self.filter[-1](state)  # shape: BN X 16    BN x 16 吧
        att_weight = self.att_func(state)  # shape: BN X 1  42 x 1
        y = (att_weight * y).view(batch_size, num_node, -1)

        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[bb, mask[bb], :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[bb, :, :], dim=0)]

        score = torch.stack(score)

        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


"""

ModuleList( 70 x 15
  (0): Linear(in_features=1050, out_features=128, bias=True) 
              128 x 15
  (1): Linear(in_features=1920, out_features=128, bias=True)
  (2): Linear(in_features=1920, out_features=128, bias=True)
  (3): Linear(in_features=1920, out_features=128, bias=True)
  (4): Linear(in_features=1920, out_features=128, bias=True)
  (5): Linear(in_features=1920, out_features=128, bias=True)
  (6): Linear(in_features=1920, out_features=128, bias=True)
  
  #最后一层输出层，输出16个分类
  (7): Linear(in_features=128, out_features=16, bias=True)
)

# 滤波器 输入的是T 不是单独的Ritz_value^long_scale  
ModuleList(
  (0): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (1): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (2): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (3): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (4): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (5): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
  (6): Sequential(
    (0): Linear(in_features=2000, out_features=4096, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Linear(in_features=4096, out_features=2000, bias=True)
  )
)

attention layer 

Sequential(
  (0): Linear(in_features=128, out_features=1, bias=True)
  (1): Sigmoid()
)

"""
