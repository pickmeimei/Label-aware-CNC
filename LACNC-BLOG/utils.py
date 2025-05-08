import os
import torch
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
# import matplotlib.pyplot as plt
from sklearn import metrics
from models import *
from layers import aggregator_lookup
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix


def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def cal_f1_score_tail(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1



def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0   
            if shuffle:
                idx = np.random.permutation(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes


def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, feature, labels, idx, cal_f1=False, is_social_net=False):
    embs = emb_model(idx, adj, feature)
    preds = cly_model(embs)
    if is_social_net:
        labels_idx = torch.argmax(labels[idx], dim=1)
        cly_loss = F.cross_entropy(preds, labels_idx)   
    else:
        cly_loss = F.multilabel_soft_margin_loss(preds, labels[idx])
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets


def evaluate(emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole





def evaluate_tail(emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False,A_head_list=None,A_tail_list=None):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    targets_whole_tail = targets_whole[A_tail_list, :]
    x = np.vstack(preds)[A_tail_list, :]
    tail_micro_f1, tail_macro_f1 = cal_f1_score_tail(targets_whole_tail, x)
    # micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))
    targets_whole_head = targets_whole[A_head_list, :]
    x1 = np.vstack(preds)[A_head_list, :]
    head_micro_f1, head_macro_f1 = cal_f1_score_tail(targets_whole_head, x1)


    return cly_loss, tail_micro_f1, tail_macro_f1, head_micro_f1, head_macro_f1, embs_whole, targets_whole


def get_split(labels, seed):
    idx_tot = np.arange(labels.shape[0])#shape取长度，arange返回一个有终点和起点的固定步长的排列（可理解为一个等差数组）
    np.random.seed(seed)
    np.random.shuffle(idx_tot)

    return idx_tot


def make_adjacency(G, max_degree, seed):
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(np.int64)
    np.random.seed(seed)
    for node in all_nodes:
        a=G[node].keys()
        a=list(a)
        neibs = np.array(a)
        # neibs = np.array(G.neighbors(node))
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels


def search_s_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #将单向边的反向边添加进图中，从而使邻接矩阵对称
    adj_dense = np.array(adj.todense()) #todense转换为正常矩阵，正常的邻接矩阵

    # 找到其长尾节点及邻居再赋值给adj_dense
    A_num = np.sum(adj_dense,1)  #计算节点的度
    A_zero_list = []
    A_tail_list = []
    A_head_list = []
    A_nei1_list = []
    A_nei2_list = []
    A_positive_list = []
    Ah_positive_list = []
    a = {}
    b = {}
    c = {} #尾的正样本对
    cn = {} #尾的负样本对
    a1 = {}
    b1 = {}
    c1 = {} #头的正对
    cn1 = {} #头的负对字典
    # c是正对的邻居字典尾
    # c1是头的正对字典
    nei1_num = 0
    nei2_num = 0
    neih1_num = 0
    neih2_num = 0
    zero_num = 0
    tail_num = 0
    head_num = 0
    for i in range(adj_dense.shape[0]):
        if A_num[i] == 0:
            zero_num = zero_num + 1
            A_zero_list.append(i)
        elif A_num[i] > 0 and A_num[i] < 9:
            tail_num += 1
            A_tail_list.append(i)
        else:
            head_num += 1
            A_head_list.append(i)
    pass
    for i in A_tail_list:
        A_nei1_list = []
        A_nei2_list = []
        for j in range(adj_dense.shape[0]):
            if adj_dense[i][j] != 0 and adj_dense[j][i] != 0:
                if j not in A_nei1_list:
                    nei1_num = nei1_num + 1
                    A_nei1_list.append(j)
                    # a[i] = A_nei1_list
                    for k in range(adj_dense.shape[0]):
                        if adj_dense[j][k] != 0 and adj_dense[k][j] != 0 and k != i:
                            if k not in A_nei2_list:
                                nei2_num = nei2_num + 1
                                A_nei2_list.append(k)



        a[i] = A_nei1_list
        b[i] = A_nei2_list

    Y = np.array(labels)
    for m in b:
        A_positive_list = []
        A_negetive_list = []
        for n in b[m]:
            if (Y[m] == Y[n]).all():
                if n not in A_positive_list:
                    A_positive_list.append(n)
                    # c[m] =A_positive_list
                    # adj_dense[m][n] = 1
                    # adj_dense[n][m] = 1


            else:
                if n not in A_negetive_list:
                    A_negetive_list.append(n)

        c[m] = A_positive_list
        cn[m] = A_negetive_list

        # 头节点不进行添边
    for i in A_head_list:
        Ah_nei1_list = []
        Ah_nei2_list = []
        for j in range(adj_dense.shape[0]):
            if adj_dense[i][j] != 0 and adj_dense[j][i] != 0:
                if j not in Ah_nei1_list:
                    neih1_num = neih1_num + 1
                    Ah_nei1_list.append(j)
                    # a[i] = A_nei1_list
                    for k in range(adj_dense.shape[0]):
                        if adj_dense[j][k] != 0 and adj_dense[k][j] != 0 and k != i:
                            if k not in Ah_nei2_list:
                                neih2_num = neih2_num + 1
                                Ah_nei2_list.append(k)

        a1[i] = Ah_nei1_list
        b1[i] = Ah_nei2_list

    Y = np.array(labels)
    for m in b1:
        Ah_positive_list = []
        Ah_negetive_list = []
        for n in b1[m]:
            if (Y[m] == Y[n]).all():
                if n not in Ah_positive_list:
                    Ah_positive_list.append(n)
                    # c[m] =A_positive_list
                    # adj_dense[m][n] = 1
                    # adj_dense[n][m] = 1

            else:
                if n not in Ah_negetive_list:
                    Ah_negetive_list.append(n)

        c1[m] = Ah_positive_list
        cn1[m] = Ah_negetive_list


    edges = np.vstack(np.where(adj_dense)).T #哪两个节点之间有边，一共多少条边
    Graph = nx.from_edgelist(edges)   #从边列表返回一个图
    adj = make_adjacency(Graph, 128, seed) #邻接矩阵adj的构建
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    #Y =np.array(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device), c,c1,cn,cn1,adj_dense,A_num



def search_t_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    #找到目标域的尾节点的二跳节点
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #将单向边的反向边添加进图中，从而使邻接矩阵对称
    adj_dense = np.array(adj.todense()) #todense转换为正常矩阵，正常的邻接矩阵

    # 找到其长尾节点及邻居再赋值给adj_dense
    A_num = np.sum(adj_dense,1)  #计算节点的度
    #计算节点平均的度
    A_zero_list = []
    A_tail_list = []
    A_head_list = []
    A_nei1_list = []
    A_nei2_list = []

    a = {}
    b = {}
    #b是目标域尾的二跳节点
    c={}
    #c是目标域二跳的正对
    cn={}
    #cn是目标域二跳的负对

    nei1_num = 0
    nei2_num = 0
    zero_num = 0
    tail_num = 0
    head_num = 0
    for i in range(adj_dense.shape[0]):
        if A_num[i] == 0:
            zero_num = zero_num + 1
            A_zero_list.append(i)
        elif A_num[i] > 0 and A_num[i] < 9:
            tail_num += 1
            A_tail_list.append(i)
        else:
            head_num += 1
            A_head_list.append(i)

    for i in A_tail_list:
        A_nei1_list = []
        A_nei2_list = []
        for j in range(adj_dense.shape[0]):
            if adj_dense[i][j] != 0 and adj_dense[j][i] != 0:
                if j not in A_nei1_list:
                    nei1_num = nei1_num + 1
                    A_nei1_list.append(j)
                    # a[i] = A_nei1_list
                    for k in range(adj_dense.shape[0]):
                        if adj_dense[j][k] != 0 and adj_dense[k][j] != 0 and k != i:
                            if k not in A_nei2_list:
                                nei2_num = nei2_num + 1
                                A_nei2_list.append(k)



        a[i] = A_nei1_list
        b[i] = A_nei2_list    #找到目标域的二跳节点

    Y = np.array(labels)
    for m in b:
        A_positive_list = []
        A_negetive_list = []
        for n in b[m]:
            if (Y[m] == Y[n]).all():
                if n not in A_positive_list:
                    A_positive_list.append(n)
                    # c[m] =A_positive_list
                    # adj_dense[m][n] = 1
                    # adj_dense[n][m] = 1
            else:
                if n not in A_negetive_list:
                    A_negetive_list.append(n)


        c[m] = A_positive_list
        cn[m] = A_negetive_list



    edges = np.vstack(np.where(adj_dense)).T #哪两个节点之间有边，一共多少条边
    Graph = nx.from_edgelist(edges)   #从边列表返回一个图
    adj = make_adjacency(Graph, 128, seed) #邻接矩阵adj的构建
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    #Y =np.array(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device),b,adj_dense,c,cn,A_num



def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #将单向边的反向边添加进图中，从而使邻接矩阵对称
    adj_dense = np.array(adj.todense()) #todense转换为正常矩阵，正常的邻接矩阵


    edges = np.vstack(np.where(adj_dense)).T #哪两个节点之间有边，一共多少条边
    Graph = nx.from_edgelist(edges)   #从边列表返回一个图
    adj = make_adjacency(Graph, 128, seed) #邻接矩阵adj的构建
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)


def load_data_s(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False,adj_s_dense=None):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #将单向边的反向边添加进图中，从而使邻接矩阵对称
    adj_dense = np.array(adj.todense()) #todense转换为正常矩阵，正常的邻接矩阵
    adj_dense =adj_s_dense
    edges = np.vstack(np.where(adj_s_dense)).T #哪两个节点之间有边，一共多少条边
    Graph = nx.from_edgelist(edges)   #从边列表返回一个图
    adj = make_adjacency(Graph, 128, seed) #邻接矩阵adj的构建
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)


def load_data_t(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False,adj_t_dense=None):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   #将单向边的反向边添加进图中，从而使邻接矩阵对称
    adj_dense = np.array(adj.todense()) #todense转换为正常矩阵，正常的邻接矩阵
    adj_dense =adj_t_dense
    edges = np.vstack(np.where(adj_t_dense)).T #哪两个节点之间有边，一共多少条边
    Graph = nx.from_edgelist(edges)   #从边列表返回一个图
    adj = make_adjacency(Graph, 128, seed) #邻接矩阵adj的构建
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)






class Net3(nn.Module):
    def __init__(self, n1, n2,n3,n4,n5,n6,n7,n8,n9):
        super(Net3, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.n6 = n6
        self.n7 = n7
        self.n8 = n8
        self.n9 = n9

        self.l1 = nn.Linear(n1, n2)
        self.l2 = nn.Linear(n2, n3)
        self.l3 = nn.Linear(n3, n4)
        self.l4 = nn.Linear(n4, n5)
        self.l5 = nn.Linear(n5, n6)
        self.l6 = nn.Linear(n6, n7)
        self.l7 = nn.Linear(n7, n8)
        self.l8 = nn.Linear(n8, n9)
        self.l9 = nn.Linear(n9, 1)



        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, input_x):
        input_x = torch.relu(self.l1(input_x))
        input_x = torch.relu(self.l2(input_x))
        input_x = torch.relu(self.l3(input_x))
        input_x = torch.relu(self.l4(input_x))
        input_x = torch.relu(self.l5(input_x))
        input_x = torch.relu(self.l6(input_x))
        input_x = torch.relu(self.l7(input_x))
        input_x = torch.relu(self.l8(input_x))

        output = torch.sigmoid(self.l9(input_x))
        return output


class Net2(nn.Module):
    def __init__(self, n1, n2,n3,n4,n5):
        super(Net2, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4=n4
        self.n5=n5

        self.l1 = nn.Linear(n1, n2)
        self.l2 = nn.Linear(n2, n3)
        self.l3 = nn.Linear(n3, n4)
        self.l4=nn.Linear(n4,n5)
        self.l5=nn.Linear(n5,1)


        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, input_x):
        input_x = torch.relu(self.l1(input_x))
        input_x = torch.tanh(self.l2(input_x))
        input_x = torch.relu(self.l3(input_x))
        input_x=torch.tanh(self.l4(input_x))
        output = torch.sigmoid(self.l5(input_x))
        return output





class Pseudo_Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k # 聚类的个数
        self.temperature = 0.1 #
        self.criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
        self.eps = 1e-8  # 一个很小的正数，用于避免除零或者对数零

    def forward(self, x):
        # 前向传播，计算损失值
        cluster_id, cluster_center = self.k_means(x) # 调用k-means算法，得到聚类标签和聚类中心
        pseudo_label = self.assign_label(cluster_id) # 根据聚类标签，分配伪标签

        x = x / (x.norm(dim=1, keepdim=True) + self.eps)  # 对数据点进行归一化处理
        cluster_center = cluster_center / (cluster_center.norm(dim=1, keepdim=True) + self.eps)  # 对聚类中心进行归一化处理

        logits = torch.matmul(x, cluster_center.t()) / (self.temperature + self.eps)# 计算数据点和聚类中心之间的相似度

        # # 增加一个负样本采样的步骤，从不同类的聚类中心中随机选择一个作为负样本，并将其与正样本拼接起来
        # neg_index = torch.randint(0, self.k, size=(x.size(0),))  # 随机生成一个负样本索引向量
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义一个设备对象
        # neg_index = neg_index.to(device)  # 将 neg_index 转移到 GPU 上
        # pseudo_label = pseudo_label.to(device)  # 将 pseudo_label 转移到 GPU 上
        # neg_index[pseudo_label == neg_index] = (neg_index[pseudo_label == neg_index] + 1) % self.k  # 避免负样本索引与正样本索引相同
        # neg_center = cluster_center[neg_index]  # 根据负样本索引选择对应的聚类中心作为负样本
        # neg_logits = torch.sum(x * neg_center, dim=1, keepdim=True) / self.temperature  # 计算数据点和负样本之间的相似度，并除以温度参数
        # logits = torch.cat([logits, neg_logits], dim=0)  # 将正样本和负样本的相似度拼接起来
        #
        # # 增加一个新的伪标签向量，用于表示正样本和负样本的分类，其中正样本对应原来的聚类标签，负样本对应最后一列
        # new_pseudo_label = torch.cat([pseudo_label, torch.full((x.size(0),), self.k, dtype=torch.long).to(device)],dim=0)# 将原来的伪标签和最后一列的索引拼接起来
        # new_pseudo_label = new_pseudo_label.to(device)
        #
        # loss = self.criterion(logits, new_pseudo_label)  # 计算交叉熵损失

        mask = torch.isnan(logits)  # 得到一个布尔型的张量，表示哪些元素是 nan
        logits = logits.masked_fill(mask, 0)  # 用 0 来替换 nan 值

        if has_invalid_values(logits):
            print("logits数据不合理")
        if has_invalid_values(pseudo_label):
            print("pseudo_label数据不合理")

        loss = self.criterion(logits, pseudo_label) # 计算交叉熵损失





        return loss

    def k_means(self, x):
        cluster_id, cluster_center = self.k_mea(x, self.k)
        return cluster_id, cluster_center

    def assign_label(self, cluster_id):
        # 根据聚类标签，分配伪标签
        pseudo_label = cluster_id  # 直接使用聚类标签作为伪标签
        return pseudo_label

    def k_mea(self, x, num_clusters):
        # Step 1: 初始化聚类中心
        indices = torch.randperm(x.size(0))[:num_clusters]  # 随机选择初始聚类中心
        centroids = x[indices]

        for _ in range(100):  # 这里我们简单地设置最大迭代次数为 100
            # Step 2: 计算每个数据点到各个聚类中心的距离，并为每个数据点分配最近的聚类中心
            dists = torch.cdist(x, centroids)  # 计算距离
            cluster_ids = dists.argmin(dim=1)  # 选择最近的聚类中心

            # Step 3: 对于每个聚类，计算其数据点的平均值并更新聚类中心
            new_centroids = torch.stack([x[cluster_ids == i].mean(dim=0) for i in range(num_clusters)])

            # Step 4: 如果聚类中心没有发生显著变化，那么就停止迭代
            if torch.allclose(centroids, new_centroids, rtol=1e-4):
                break

            centroids = new_centroids

        return cluster_ids, centroids


def has_invalid_values(tensor):
    # 检查是否存在 NaN 或无穷大值
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True

    # 检查是否存在无穷小值
    if torch.isneginf(tensor).any():
        return True

    # 检查是否存在无穷大值
    if torch.isposinf(tensor).any():
        return True

    return False




class Contrastive_Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k # 聚类的个数
        self.temperature = 0.1 # 温度参数，用于调节相似度的尺度
        self.criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    def forward(self, x):
        # 前向传播，计算损失值
        cluster_id, cluster_center = self.k_means(x) # 调用k-means算法，得到聚类标签和聚类中心
        pseudo_label = self.assign_label(cluster_id) # 根据聚类标签，分配伪标签
        logits = torch.matmul(x, cluster_center.t()) / self.temperature # 计算数据点和聚类中心之间的相似度，并除以温度参数
        loss = self.criterion(logits, pseudo_label) # 计算交叉熵损失
        return loss

    def k_means(self, x):
        cluster_id, cluster_center = self.k_mea(x, self.k)
        return cluster_id, cluster_center

    def assign_label(self, cluster_id):
        # 根据聚类标签，分配伪标签
        pseudo_label = cluster_id  # 直接使用聚类标签作为伪标签
        return pseudo_label

    def k_mea(self, x, num_clusters):
        # Step 1: 初始化聚类中心，使用 k-means++ 方法
        centroids = self.k_means_plus_plus(x, num_clusters)

        for _ in range(100):  # 这里我们简单地设置最大迭代次数为 100
            # Step 2: 计算每个数据点到各个聚类中心的距离，并为每个数据点分配最近的聚类中心
            dists = torch.cdist(x, centroids)  # 计算距离
            cluster_ids = dists.argmin(dim=1)  # 选择最近的聚类中心

            # Step 3: 对于每个聚类，计算其数据点的平均值并更新聚类中心
            new_centroids = torch.stack([x[cluster_ids == i].mean(dim=0) for i in range(num_clusters)])

            # Step 4: 如果聚类中心没有发生显著变化，那么就停止迭代
            if torch.allclose(centroids, new_centroids, rtol=1e-4):
                break

            centroids = new_centroids

        return cluster_ids, centroids

    def k_means_plus_plus(self, x, num_clusters):
        # k-means++ 初始化方法，参考 https://en.wikipedia.org/wiki/K-means%2B%2B
        n = x.size(0) # 数据点的个数
        centroids = [] # 聚类中心的列表
        indices = torch.randperm(n) # 随机打乱数据点的顺序
        centroids.append(x[indices[0]]) # 随机选择第一个聚类中心
        for _ in range(1, num_clusters): # 循环选择剩余的聚类中心
            dists = torch.cdist(x, torch.stack(centroids)) # 计算数据点和已选聚类中心之间的距离
            min_dists = dists.min(dim=1).values # 计算数据点和最近的聚类中心之间的距离
            probs = min_dists / min_dists.sum() # 计算数据点被选为下一个聚类中心的概率，距离越大，概率越大
            index = torch.multinomial(probs, 1).item() # 按照概率分布，随机选择一个数据点作为下一个聚类中心
            centroids.append(x[index]) # 将该数据点加入到聚类中心的列表中

        return torch.stack(centroids) # 返回聚类中心的张量


    # # 中心正则化
        # dist_matrix = torch.cdist(cluster_center, cluster_center)  # 计算聚类中心两两之间的距离矩阵，是一个 (k, k) 的张量
        # sim_matrix = 1 - dist_matrix  # 计算聚类中心两两之间的相似度矩阵，也是一个 (k, k) 的张量
        # center_reg = torch.mean(sim_matrix)  # 计算聚类中心之间的平均相似度
        # # center_reg = torch.mean(1 - torch.cosine_similarity(cluster_center))
        # loss += 0.1 * center_reg
        #
        # # 类内压缩
        # center_vecs = cluster_center[pseudo_label]
        # in_var = torch.mean(torch.norm(x - center_vecs, dim=1))
        # loss += 0.1 * in_var
        #
        # # 中心方差正则化
        # c_var = torch.mean(torch.norm(cluster_center, dim=1))
        # loss += 0.1 * c_var