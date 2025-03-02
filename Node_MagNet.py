# 数据保存
import pickle as pk
# 优化器
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
import torch
# 数据集
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS, Planetoid

# internal files
from utils.Citation import *

from layer.sparse_magnet import *

from utils.preprocess import geometric_dataset_sparse, load_syn
from utils.save_settings import write_log

# hermitian矩阵分解，但是没有被使用
from utils.hermitian import hermitian_decomp_sparse

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:0")


def parse_args():
    parser = argparse.ArgumentParser(description="MagNet Conv (sparse version)")
    parser.add_argument('--log_root', type=str, default='../logs/',
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test',
                        help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/data/tmp/',
                        help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Texas', help='data set selection')

    parser.add_argument('--epochs', type=int, default=3000, help='Number of (maximal) training epochs.')
    # (q)
    parser.add_argument('--q', type=float, default=0.15, help='q value for the phase matrix')
    # 方向强度参数是beta吗
    parser.add_argument('--p_q', type=float, default=0.95, help='Direction strength, from 0.5 to 1.')
    parser.add_argument('--p_inter', type=float, default=0.1, help='Inter-cluster edge probabilities.')
    parser.add_argument('--method_name', type=str, default='Magnet', help='method name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for training testing split/random graph generation.')

    # 切比雪夫多项式刻度
    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--num_filter', type=int, default=512, help='num of filters')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--new_setting', '-NS', action='store_true', help='Whether not to load best settings.')


    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularizer')

    parser.add_argument('-activation', '-a', action='store_true', help='if use activation function')

    parser.add_argument('--randomseed', type=int, default=-1, help='if set random seed in training')
    return parser.parse_args()


###################################################################################################################################################################################
# 数据处理
# 这个函数实际上是将切比雪夫多项式截断的滤波器转换成tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """
sparse_mx.tocoo()是一个Scipy稀疏矩阵的方法，它将稀疏矩阵转换为COO（Coordinate List）格式。

COO格式是一种常见的稀疏矩阵表示方法，在该表示中，每个非零元素都用三个数组来存储：行索引、列索引和元素值。这些数组可以轻松地与其他数据结构进行交互，比如NumPy数组。

使用sparse_mx.tocoo()方法将稀疏矩阵转换为COO格式后，可以通过访问属性.row、.col和.data获取其行、列和非零元素的数组，这些数组可以用于进一步处理稀疏矩阵。
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # torch.from_numpy(ndarray) → Tensor
    """
    np.vstack是一个NumPy库函数，用于将给定的数组垂直堆叠成一个新的数组。
具体而言，np.vstack函数将多个数组按照垂直方向（沿着行的方向）堆叠起来，生成一个新的数组。该函数的输入参数为一个元组或列表，其中包含要进行垂直堆叠的数组，它们的维度在水平方向上应该相同。输出结果是一个新的数组，其中每个输入数组在垂直方向上被堆叠在一起。
    """
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # numpy和tensor是需要进行转换的
    return torch.sparse.FloatTensor(indices, values, shape)


###########################################################################################################################3


def main(args):
    # 这是对于人造数据集来说的
    if args.randomseed > 0:
        # 手工设置的随机数种子
        torch.manual_seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H-%M-%S')

    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) == False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')

    # telegram/telegram WebKB/Cornell WebKB/Wisconsin WebKB/Texas cora_ml/ citeseer/  这是dataset的参数值
    # --epochs 3000 --lr 0.002 --num_filter 16 --q 0.05 --log_path telegram_magnet --dataset telegram/telegram  --K 1  --layer 2 --dropout 0.5
    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    # 这个load_func 是pyg提供的数据集处理方法
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
    # elif load_func == 'Pubmed':
    #     load_func = Planetoid
    else:
        # 处理手工生成的数据集
        load_func = load_syn

    _file_ = args.data_path + args.dataset + '/data' + str(args.q) + '_' + str(args.K) + '_sparse.pk'
    if os.path.isfile(_file_):
        # data是一个字典
        data = pk.load(open(_file_, 'rb'))
        # WebKB的cornell数据集data0.25_2_sparse.pk的数据文件是183*183
        # 这个L应该是hermitian_decomp_sparse函数生成的，如果之前有保存。pk文件那么就会从。pk文件中读取
        L = data['L']
        # 数据预处理
        X, label, train_mask, val_mask, test_mask = geometric_dataset_sparse(args.q, args.K,
                                                                             root=args.data_path + args.dataset,
                                                                             subset=subset,
                                                                             dataset=load_func, load_only=True,
                                                                             save_pk=False)
    else:
        X, label, train_mask, val_mask, test_mask, L = geometric_dataset_sparse(args.q, args.K,
                                                                                root=args.data_path + args.dataset,
                                                                                subset=subset,
                                                                                dataset=load_func, load_only=False,
                                                                                save_pk=True)

    # normalize label, the minimum should be 0 as class index
    # 返回数组的最小值或沿轴方向的最小值
    _label_ = label - np.amin(label)
    # 最终输出的分类类别
    cluster_dim = np.amax(_label_) + 1

    # convert dense laplacian to sparse matrix 这个laplacian矩阵就是磁化的
    # 他把实部和虚部分离了  在Python中，复数可以用实数和虚数的形式表示为a+bj，其中a是实部，b是虚部，j是虚数单位。在列表L中，假设L[i]是一个复数，那么L[i].imag表示该复数的虚部值（即b值）。例如，如果L[i] = 3+4j，则L[i].imag将返回4.0。
    # 实数
    L_img = []
    # 虚数
    L_real = []
    for i in range(len(L)):
        L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))  # 将虚部转换成tensor
        L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))  # 将实部扎unhuan为tensor

    label = torch.from_numpy(_label_[np.newaxis]).to(device)  # 将分类标签也转换成tensor

    # 这个X是什么？
    # 这个X是什么？
    X = X*(np.sqrt(2)/2 + np.sqrt(2)*1j/2)
    # X从返回结果看是节点特征矩阵：N x num_features
    X_img = torch.FloatTensor(X.imag).to(device)

    X_real = torch.FloatTensor(X.real).to(device)

    # 这是分类损失
    criterion = nn.NLLLoss()  # nn.NLLLoss()是PyTorch中的一个损失函数，全称是Negative Log Likelihood Loss。在分类问题中，我们通常将输出层的激活值视为预测的类别分布，NLLLoss计算的是预测分布与真实分布之间的差异（也就是交叉熵），并将其作为模型训练的目标函数。因此，NLLLoss通常用于多分类问题，并且要求网络最后一层使用log_softmax()作为激活函数。在使用NLLLoss时，网络的输出应该是每个类别的对数概率值，而不是原始的概率值。
    # 数据切片，论文中切了10个
    splits = train_mask.shape[1]  # 相当于数据取了10次，每次按照60%，20%，20%的比例划分训练集，验证集和测试集

    if len(test_mask.shape) == 1:
        # data.test_mask = test_mask.unsqueeze(1).repeat(1, splits)
        test_mask = np.repeat(test_mask[:, np.newaxis], splits, 1)
    # 这个4是什么
    results = np.zeros((splits, 4))
    # 对每个划分的数据
    for split in range(splits):
        log_str_full = ''

        # 模型初始化    L_real:拉普拉斯矩阵的实数部分   L_img:拉普拉斯矩阵的虚数部分
        model = ChebNet(X_real.size(-1), L_real, L_img, K=args.K, label_dim=cluster_dim, layer=args.layer,
                        activation=True, num_filter=args.num_filter, dropout=args.dropout).to(device)

        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_test_acc = 0.0
        # 训练集索引   这个其实就是他分十次取数据，每次都取相同的数据，但是每次取的时候是按照60%2020%的比例划分训练集,，测试集和验证集的
        train_index = train_mask[:,
                      split]  # 这段代码是在对一个bool型的二维矩阵train_mask进行操作。train_mask中每个元素代表了数据集中对应位置的样本是否被标记为训练集（True表示是，False表示不是）。split是一个整数，表示第split个训练集。train_mask[:,split]则取出了train_mask的第split列，即此时train_index就是一个一维的bool型向量，它表示了所有在第split个训练集中的样本。
        # 验证集索引
        val_index = val_mask[:, split]
        # 测试级索引
        test_index = test_mask[:, split]

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0  # 若长期出现精度没有提高的情况就停止训练
        # 记录开始时间

        # 开始训练
        for epoch in range(args.epochs):
            start_time = time.time()

            ####################
            # Train
            ####################
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(train_index)

            model.train()
            preds = model(X_real, X_img)
            train_loss = criterion(preds[:, :, train_index], label[:, train_index].long())
            pred_label = preds.max(dim=1)[1]
            train_acc = 1.0 * ((pred_label[:, train_index] == label[:, train_index])).sum().detach().item() / count

            train_loss.backward()  # 这是计算梯度
            opt.step()  # opt.step() 是 PyTorch 中优化器的一个方法调用。在训练神经网络时，需要对模型参数进行更新，以使损失函数最小化。优化器是用于执行此操作的工具之一。opt.step() 方法会根据损失函数的梯度来更新模型的参数。换句话说，它将通过计算每个参数的梯度并将其与学习率相乘来计算新的参数值。然后，这些新的参数值将被用于下一轮的反向传播和梯度计算。通过多次迭代和更新参数，神经网络将能够逐渐提高对训练数据的拟合程度，并在测试数据上获得更好的性能。
            opt.zero_grad()  # 梯度清零应该得在反向传播之后把
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

            test_loss = criterion(preds[:, :, val_index], label[:, val_index].long())
            test_acc = 1.0 * ((pred_label[:, val_index] == label[:, val_index])).sum().detach().item() / count

            outstrval = ' Test loss:, %.6f, acc:, %.3f,' % (test_loss.detach().item(), test_acc)

            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d ,/, %d ,epoch," % (epoch, args.epochs)) + outstrtrain + outstrval + duration
            log_str_full += log_str + '\n'
            # print(log_str)

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
            if early_stopping > 500 or epoch == (args.epochs - 1):
                torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
                break

        write_log(vars(args), log_path)
        # 记录结束时间
        end_time = time.perf_counter()

        # 计算并打印训练时间

        #print(f'Training time: {training_time:.2f} seconds')

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + '/model' + str(split) + '.t7'))
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
    print(args)
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