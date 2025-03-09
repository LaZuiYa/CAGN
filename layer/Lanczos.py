import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import eig, eigh
from pylanczos import PyLanczos
from scipy.sparse import coo_matrix
from scipy.linalg import qr

import pickle as pk
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh

cuda_device = 0
device = torch.device("cuda:0")
EPS = float(np.finfo(np.float32).eps)

rng = np.random.default_rng(seed=1234)
torch.manual_seed(1234)





def lanczos_approxmiate_py(step, A):
    engine = PyLanczos(A, True, step)
    eigenvalues, eigenvectors = engine.run()

    # print(A)
    print(np.mean(np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T.conj()) - A)))

    return coo_matrix(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T.conj()))


def lanczos_approxmiate_py_Tensor(step, A, datapath=None):
    # if os.path.isfile(datapath + str(step) + "LanczosWithChebPoly.pk"):
    #     dataset = pk.load(open(datapath + str(step)  + "LanczosWithChebPoly.pk",'rb'))
    #     return dataset["eigenvalue"],dataset["eigvector"]
    engine = PyLanczos(A, True, step)

    eigenvalues, eigenvectors = engine.run()

    # print(A)
    # print(np.mean(np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T.conj()) - A)))
    dataset = {}

    dataset["eigenvalue"] = eigenvalues
    dataset["eigvector"] = eigenvectors

    pk.dump(dataset, open(datapath + str(step) + "LanczosWithChebPoly.pk", 'wb'))
    return eigenvalues, eigenvectors




def restart_lanczos_approxmiate__Tensor(step, H, datapath=None):
    dataset = {}
    # if os.path.isfile(datapath + str(step) + "222.pk"):
    #     dataset = pk.load(open(datapath + str(step) + "222.pk", 'rb'))
    #     print("files exist")
    #     return dataset["evals"],dataset["evecs"]
    # r,v = eigh(H.todense())
    def reorthogonalize(V):
        Q, _ = qr(V, mode='economic')
        return Q
    # 初始向量
    n = H.shape[0]
    v0 = np.random.rand(n)
    v0 /= np.linalg.norm(v0)

    # 迭代求解（伪代码逻辑）
    for i in range(20):
        # 调用 eigsh 并获取部分结果

        evals, evecs = eigsh(
            H, k=step, v0=v0,tol=1e-8,
            which='LM', return_eigenvectors=True
        )

        evecs = reorthogonalize(evecs)
        v0 = evecs[:, -1]


    # evals, evecs = eigsh(H, k=step, which='LM', return_eigenvectors=True)
    dataset["evals"] = evals
    dataset["evecs"] = evecs


    # pk.dump(dataset, open(datapath + str(step) + "222.pk", 'wb'))
    return evals, evecs


def restart_lanczos_approxmiate(step, H, datapath=None, q=0):
    dataset = {}
    if os.path.isfile(datapath + str(step) + "_" + str(q) + "LanczosWithChebPoly.pk"):
        dataset = pk.load(open(datapath + str(step) + "_" + str(q) + "LanczosWithChebPoly.pk", 'rb'))

        return dataset["eigenvalue"].reshape(1, step).astype(np.float32), dataset["eigvector"].astype(np.complex64)

    eigvalue, eigvector = eigsh(H, k=step, which='LM', return_eigenvectors=True)

    eigvector = np.array(eigvector)
    dataset["eigenvalue"] = eigvalue
    dataset["eigvector"] = eigvector
    print(np.mean(np.abs(eigvector.dot(np.diag(eigvalue)).dot(eigvector.conj().T) - H)))

    pk.dump(dataset, open(datapath + str(step) + "_" + str(q) + "LanczosWithChebPoly.pk", 'wb'))
    return eigvalue.reshape(1, step).astype(np.float32), eigvector.astype(np.complex64)

def restart_lanczos_approxmiate_Tensor(step, H, datapath=None):
    dataset = {}

        # 调用 eigsh 并获取部分结果

    evals, evecs = eigsh(
            H, k=step, tol=1e-8,
            which='SM', return_eigenvectors=True
        )
    # evals, evecs = eigsh(H, k=step, which='LM', return_eigenvectors=True)
    dataset["evals"] = evals
    dataset["evecs"] = evecs


    # pk.dump(dataset, open(datapath + str(step) + "222.pk", 'wb'))
    return evals, evecs