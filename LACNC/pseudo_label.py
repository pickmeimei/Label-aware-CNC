import torch
from torch_cluster import k_means
import torch.nn as nn

# def cluster(x, k):
#     cluster_id, cluster_center = k_means(x, k)
#     return cluster_id, cluster_center
#
#
# def assign_label(cluster_id):
#     pseudo_label = torch.unique(cluster_id, sorted=True)
#     return pseudo_label
#
# def compute_loss(x, cluster_center, pseudo_label):
#     criterion = nn.CrossEntropyLoss()
#     logits = torch.matmul(x, cluster_center.t())
#     loss = criterion(logits, pseudo_label)
#     return loss



class Pseudo_Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k # 聚类的个数
        self.criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    def forward(self, x):
        # 前向传播，计算损失值
        cluster_id, cluster_center = self.k_means(x) # 调用k-means算法，得到聚类标签和聚类中心
        pseudo_label = self.assign_label(cluster_id) # 根据聚类标签，分配伪标签
        logits = torch.matmul(x, cluster_center.t()) # 计算数据点和聚类中心之间的相似度
        loss = self.criterion(logits, pseudo_label) # 计算交叉熵损失
        return loss

    def k_means(self, x):
        # 实现k-means算法，返回聚类标签和聚类中心
        # 这里可以根据你的需要，使用不同的实现方式
        # 例如，你可以使用torch-cluster库中的k_means函数
        # 也可以自己编写k-means算法的逻辑
        cluster_id, cluster_center = torch_cluster.k_means(x, self.k) # 使用torch-cluster库
        return cluster_id, cluster_center

    def assign_label(self, cluster_id):
        # 根据聚类标签，分配伪标签
        pseudo_label = torch.unique(cluster_id, sorted=True) # 对聚类标签去重并排序，得到伪标签
        return pseudo_label

