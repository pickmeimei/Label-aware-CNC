import torch
import torch.nn as nn
import torch.nn.functional as F
from AdaGIn2 import GraphSAGE
import torch.nn as nn
from layers1 import GraphConvolution
import torch.nn.functional as F
import torch
# from torch_cluster import k_means
import torch.nn as nn




class Pseudo_Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k # 聚类的个数
        self.temperature = 1.0 #
        self.criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    def forward(self, x):
        # 前向传播，计算损失值
        cluster_id, cluster_center = self.k_means(x) # 调用k-means算法，得到聚类标签和聚类中心
        pseudo_label = self.assign_label(cluster_id) # 根据聚类标签，分配伪标签
        logits = torch.matmul(x, cluster_center.t()) / self.temperature # 计算数据点和聚类中心之间的相似度
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