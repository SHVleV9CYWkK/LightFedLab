import math
import cvxpy as cp
import numpy as np

from servers.fl_method_servers.fedwcp_server import FedWCPServer


class AdFedWCPServer(FedWCPServer):
    def __init__(self, clients, model, device, args):
        self.n_rounds = args['n_rounds']
        self.current_rounds = 0
        self.avg_loss_change = float('inf')
        self.last_loss = self.current_loss = 0

        self.k_min = 5
        self.k_max = 32
        self.datasets_len = {}
        self.max_dataset_len = 0
        self.min_dataset_len = float('inf')

        self.model_size = sum(p.numel() for p in model.parameters()) * 4

        for client in clients:
            self.datasets_len[client.id] = client.train_dataset_len
            if self.datasets_len[client.id] > self.max_dataset_len:
                self.max_dataset_len = self.datasets_len[client.id]
            if self.datasets_len[client.id] < self.min_dataset_len:
                self.min_dataset_len = self.datasets_len[client.id]
        self.data_volume_scale_factor = (self.k_max - self.k_min) / (
                self.max_dataset_len * 1.5 - self.min_dataset_len / 1.5)

        self.bandwidth_min = 5
        self.bandwidth_max = 100
        self.bandwidths = [10] * len(clients)
        for client, dataset_len in self.datasets_len.items():
            normalized_size = (dataset_len - self.min_dataset_len) / (self.max_dataset_len - self.min_dataset_len)
            self.bandwidths[client] = math.ceil(
                self.bandwidth_min + normalized_size * (self.bandwidth_max - self.bandwidth_min))

        self.bandwidth_scale_factor = (self.k_max - self.k_min) / (self.bandwidth_max * 1.5 - self.bandwidth_min / 1.5)

        super().__init__(clients, model, device, args)

    def k_lower_bound(self, data_volume, current_epoch, total_epochs, avg_loss_change=1):
        base_k = self.k_min + math.ceil((data_volume - self.min_dataset_len) * self.data_volume_scale_factor)

        # 考虑训练进度，越接近结束，可能希望k值越大
        progress_factor = (1 + (current_epoch / total_epochs))

        # 考虑损失变化率，损失变化小于某个阈值时，降低k值
        if avg_loss_change < 0.00015:
            loss_factor = 1.1
        else:
            loss_factor = 1

        # 计算最终的k值
        final_k = max(self.k_min, math.ceil(base_k * progress_factor * loss_factor))
        return min(self.k_max, final_k)  # 确保k值不小于最小值

    def k_upper_bound(self, bandwidth):
        result = self.k_max - math.ceil((self.bandwidth_max - bandwidth) * self.bandwidth_scale_factor)
        return max(self.k_min, min(self.k_max, result))

    @staticmethod
    def compression_rate(k):
        return 0.00395 * k + 0.07567  # 近似线性函数

    def determine_k(self, current_epoch, total_epochs, avg_loss_change=1):
        # 优化变量
        k = cp.Variable(len(self.clients), integer=True)
        # 目标函数：最小化通信成本
        objective = cp.Minimize(cp.sum((self.compression_rate(k) * self.model_size) / np.array(self.bandwidths)))
        # 约束条件列表 最大最小上下边界
        constraints = []
        # 添加带宽引起和数据量和k值上下限为额外约束
        constraints += [k[client.id] >= self.k_lower_bound(self.datasets_len[client.id],
                                                           current_epoch,
                                                           total_epochs,
                                                           avg_loss_change) for client in self.clients]  # 数据量越多，k越大
        constraints += [k[client.id] <= self.k_upper_bound(self.bandwidths[client.id]) for client in self.clients]  # 带宽越小，k越小
        # 定义和求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status in ["infeasible", "unbounded"]:
            raise Exception("Infeasible or unbounded")
        else:
            return k.value

    def abc(self):
        alpha = 0.5
        print("Calculating k...")
        if self.current_rounds == 0:
            k_list = self.determine_k(self.current_rounds, self.n_rounds)
        else:
            self.avg_loss_change = alpha * (abs(self.current_loss - self.last_loss)) + (
                        1 - alpha) * self.avg_loss_change  # EMA
            k_list = self.determine_k(self.current_rounds, self.n_rounds, self.avg_loss_change)
        print(k_list)
        for idx, k in enumerate(k_list):
            self.clients[idx].set_num_centroids(int(k))
        self.last_loss = self.current_loss

    def _init_clients(self):
        self.abc()
        super()._init_clients()

    def train(self):
        self.abc()
        super().train()
        self.current_rounds += 1

    def evaluate(self):
        result = super().evaluate()
        self.current_loss = result['loss']
        return result
