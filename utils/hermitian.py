import numpy as np
from numpy import linalg as LA
from pylanczos import PyLanczos
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import eigsh
# from src.layer.Lanczos import lanczos_approxmiate, lanczos_approxmiate_py, restart_lanczos_approxmiate__Tensor, \
#     restart_lanczos_approxmiate

# 切比雪夫多项式分解，运算成本和内存成本较高，可以用lanczos算法进行低秩近似


###########################################
####### Dense implementation ##############
###########################################
def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian[0] += np.eye(N, dtype=np.float32)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] += A
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] += 2 * np.dot(A, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]

    return multi_order_laplacian


def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
    A = 1.0 * np.array(A)
    if gcn_appr:
        A += 1.0 * np.eye(A.shape[0])

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.sum(np.array(A_sym), axis=0)
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = np.diag(d)
        A_sym = np.dot(np.dot(D, A_sym), D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array
        if norm:
            D = np.diag([1.0] * len(d))
        else:
            d = np.sum(np.array(A_sym), axis=0)  # diag of degree array
            D = np.diag(d)
        L = D - np.exp(Theta) * A_sym
    '''
    else:
        #transition matrix
        d_out = np.sum(np.array(A), axis = 1)
        d_out[d_out==0] = -1
        d_out = 1.0/d_out
        d_out[d_out<0] = 0
        D = np.diag(d_out)
        L = np.eye(len(d_out)) - np.dot(D, A)
    '''
    w, v = None, None
    if norm:
        if max_eigen == None:
            w, v = LA.eigh(L)
            L = (2.0 / np.amax(np.abs(w))) * L - np.diag([1.0] * len(A))
        else:
            L = (2.0 / max_eigen) * L - np.diag([1.0] * len(A))
            w = None
            v = None

    return L, w, v


# 除了Magnet其他都用的是非稀疏矩阵存储
def hermitian_decomp(As, q=0.25, norm=False, laplacian=True, max_eigen=None, gcn_appr=False):
    ls, ws, vs = [], [], []
    if len(As.shape) > 2:
        for i, A in enumerate(As):
            l, w, v = decomp(A, q, norm, laplacian, max_eigen, gcn_appr)
            vs.append(v)
            ws.append(w)
            ls.append(l)
    else:
        ls, ws, vs = decomp(As, q, norm, laplacian, max_eigen, gcn_appr)
    return np.array(ls), np.array(ws), np.array(vs)


###########################################
####### Sparse implementation #############
###########################################


def cheb_poly_sparse(A, K):
    # 注意这个K+1
    K += 1
    N = A.shape[0]  # [N, N] A实际上是磁化的拉普拉斯矩阵L，但是为了降低计算成本，所以将A进行归一化 即～L := 2/λmax L − I. 然后用～L代替～Λ
    # multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append(coo_matrix((np.ones(N), (np.arange(N), np.arange(N))),
                                            shape=(N, N), dtype=np.float32))
    if K == 1:  # 实际上这是T0
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:  # 这是T1
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append(
                    2.0 * A.dot(multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2])  # 正常的矩阵乘法

    return multi_order_laplacian
# def update_matrix(A):
#     # 转换为 CSR 格式，以便可以通过下标访问元素
#     A = A.tocsr()
#
#     # 存储更新后的数据
#     row, col, data = [], [], []
#
#     # 遍历所有非零元素
#     for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
#         # if i>j:
#         a_ij = A[i, j]
#         a_ji = A[j, i] if j != i else 0
#         updated_value =  a_ij - 1j*a_ji
#         # else:
#         #     a_ij = A[i, j]
#         #     a_ji = A[j, i]
#         #     updated_value =  a_ji - 1j*a_ij
#
#         # 记录 (i,j) 的更新值
#         row.append(i)
#         col.append(j)
#         data.append(updated_value)
#
#     # 构造新的稀疏矩阵，使用 COO 格式创建并转换为 CSR 格式
#     A_updated = coo_matrix((data, (row, col)), shape=A.shape).tocsr()
#
#     return A_updated

def update_matrix(A):
    # 转置矩阵获取入度信息
    A = A.tocsr()

    # 存储更新后的数据
    row, col, data = [], [], []

    # 遍历所有非零元素
    for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
        if i == j:
            row.append(i)
            col.append(j)
            data.append(1-1j)
            continue

        a_ij = A[i, j]
        a_ji = A[j, i]
        if a_ji == 1:
            updated_value =  a_ij - 1j*a_ji
            # 记录 (i,j) 的更新值
            row.append(i)
            col.append(j)
            data.append(updated_value)
        else:
            row.append(i)
            col.append(j)
            data.append(1)
            row.append(j)
            col.append(i)
            data.append(-1j)
    # 构造新的稀疏矩阵，使用 COO 格式创建并转换为 CSR 格式
    A_updated = coo_matrix((data, (row, col)), shape=A.shape).tocsr()

    return A_updated

# hermation  用scite_learn的coo_matrix来存储laplacian矩阵
def hermitian_decomp_sparse(row, col, size, q=0.25, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    if edge_weight is None:  # 无权图

        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size),
                      dtype=np.float32)  # 度数矩阵Ds
    A = update_matrix(A)
    if gcn_appr:
        A += diag  # 这就是生成普通的对角矩阵，仅对于无向图 A+D

    A_sym = 0.5 * (A + A.conj().T)  # symmetrized adjacency    As(u, v) := 1/2 (A(u, v) + A(v, u))
    # A_sym = A +diag
    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree 沿着列轴计算每个节点出度
        d[d == 0] = 1  # 这行代码是将数组d中所有等于0的元素替换为1。在Python中，表达式"d == 0"会返回一个相同形状的布尔数组，其值为True或False表示每个元素是否等于0。使用这个布尔数组作为索引，可以选择只对等于0的元素进行操作。因此，这个语句就相当于选取了d中所有为0的元素，并将它们的值改为1。避免存在孤立节点导致除0
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size))
        A_sym = D.dot(A_sym).dot(D)  # Ds^−1/2 As Ds^−1/2  归一化

    #
    if laplacian:
        if norm:  # 这是什么操作？ 哦，他为了省事实际上这个D是I单位阵
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array  #不归一化  L(q) U := Ds−H(q) = Ds−Asexp(iΘ(q)),
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - A_sym  # element-wise归一化：L(q) N := I− ( D−1/2 s AsD−1/2 s ) exp(iΘ(q)) . (1)
    if norm:
        eigenvalues, _ = eigsh(L,k=1,which='LA')
        L = (2.0 / eigenvalues.max()) * L - diag  # 直接默认你最大特征值是2   实际上是这个 ～Λ = 2/λmax Λ − I    analogous to  ̃ Λ, we define  ̃ L := 2/λmax L − I.
    # L = update_matrix(L)
    return L


def hermitian_decomp_sparse2(row, col, size, q=0.1, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array
        Theta.data = np.exp(Theta.data)
        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym)  # element-wise

    if norm:
        engine = PyLanczos(L, True, 1)
        eigenvalues, eigenvectors = engine.run()
        L = (2/2)* L - diag  # 直接默认你最大特征值是2   实际上是这个 ～Λ = 2/λmax Λ − I    analogous to  ̃ Λ, we define  ̃ L := 2/λmax L − I.
    return L

def hermitian_decomp_sparse3(row, col, size, q=0.25, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    if edge_weight is None:  # 无权图

        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size),
                      dtype=np.float32)  # 度数矩阵Ds
    A = update_matrix2(A)
    if gcn_appr:
        A += diag  # 这就是生成普通的对角矩阵，仅对于无向图 A+D

    A_sym = 0.5 * (A + A.T) + identity(size)  # symmetrized adjacency    As(u, v) := 1/2 (A(u, v) + A(v, u))

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree 沿着列轴计算每个节点出度
        d[
            d == 0] = 1  # 这行代码是将数组d中所有等于0的元素替换为1。在Python中，表达式"d == 0"会返回一个相同形状的布尔数组，其值为True或False表示每个元素是否等于0。使用这个布尔数组作为索引，可以选择只对等于0的元素进行操作。因此，这个语句就相当于选取了d中所有为0的元素，并将它们的值改为1。避免存在孤立节点导致除0
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)  # Ds^−1/2 As Ds^−1/2  归一化

    #
    if laplacian:

        # Theta =np.sqrt(2)/2 - np.sqrt(2)*1j/2
        Theta = 2*np.pi*q*1j*(A - A.T) # phase angle array  1j是虚数单位   Θ(q)(u, v) := 2πq(A(u, v) − A(v, u)) ,
        Theta.data = np.exp(Theta.data)  # 取自然数指数
        if norm:  # 这是什么操作？ 哦，他为了省事实际上这个D是I单位阵
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array  #不归一化  L(q) U := Ds−H(q) = Ds−Asexp(iΘ(q)),
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - A_sym  # element-wise归一化：L(q) N := I− ( D−1/2 s AsD−1/2 s ) exp(iΘ(q)) . (1)
        #L = D - A_sym *
    if norm:
        engine = PyLanczos(L, True, 1)
        eigenvalues, eigenvectors = engine.run()
        L = (2.0 / 2) * L - diag  # 直接默认你最大特征值是2   实际上是这个 ～Λ = 2/λmax Λ − I    analogous to  ̃ Λ, we define  ̃ L := 2/λmax L − I.
    # L = update_matrix(L)
    return L

def update_matrix2(A):
    # 转换为 CSR 格式，以便可以通过下标访问元素
    A = A.tocsr()

    # 存储更新后的数据
    row, col, data = [], [], []

    # 遍历所有非零元素
    for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
        a_ij = A[i, j]
        a_ji = A[j, i] if i != j else 0  # 对角线元素的对称元素设为 0
        updated_value =  a_ij*np.sqrt(2)/2 - 1j*a_ji*np.sqrt(2)/2

        # 记录 (i,j) 的更新值
        row.append(i)
        col.append(j)
        data.append(updated_value)

    # 构造新的稀疏矩阵，使用 COO 格式创建并转换为 CSR 格式
    A_updated = coo_matrix((data, (row, col)), shape=A.shape).tocsr()

    return A_updated