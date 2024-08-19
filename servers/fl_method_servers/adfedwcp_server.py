import math
import cvxpy as cp
from servers.fl_method_servers.fedwcp_server import FedWCPServer


class AdFedWCPServer(FedWCPServer):
    def __init__(self, clients, model, device, args):
        self.n_rounds = args['n_rounds']
        self.current_rounds = 0
        self.avg_loss_change = float('inf')
        self.last_loss = self.current_loss = 0
        self.last_k_value = None
        self.k_min = 5
        self.k_max = 32
        self.datasets_len = {}
        self.max_dataset_len = 0
        self.min_dataset_len = float('inf')

        # 获取每一层模型大小的方法
        layer_sizes = []
        for layer in model.parameters():
            layer_sizes.append(layer.numel() * 4)
        self.weight_layer_sizes = [size for idx, size in enumerate(layer_sizes) if idx % 2 == 0]
        self.bias_layer_sizes = [size for idx, size in enumerate(layer_sizes) if idx % 2 != 0]

        for client in clients:
            self.datasets_len[client.id] = client.train_dataset_len
            if self.datasets_len[client.id] > self.max_dataset_len:
                self.max_dataset_len = self.datasets_len[client.id]
            if self.datasets_len[client.id] < self.min_dataset_len:
                self.min_dataset_len = self.datasets_len[client.id]
        self.data_volume_scale_factor = (self.k_max - self.k_min) / (
                self.max_dataset_len - self.min_dataset_len)

        self.bandwidth_min = 5
        self.bandwidth_max = 100
        self.bandwidths = [10] * len(clients)
        for client, dataset_len in self.datasets_len.items():
            normalized_size = (dataset_len - self.min_dataset_len) / (self.max_dataset_len - self.min_dataset_len)
            self.bandwidths[client] = math.ceil(
                self.bandwidth_min + normalized_size * (self.bandwidth_max - self.bandwidth_min))

        self.bandwidth_scale_factor = (self.k_max - self.k_min) / (self.bandwidth_max - self.bandwidth_min)

        super().__init__(clients, model, device, args)

    def interlayers_k_constraints(self, k, current_epoch, data_volume, bandwidth, importance_weight,
                                  avg_loss_change=1.0):
        # 数据量影响下界
        adjusted_k_lower_bound = self.k_min + math.ceil(
            self.data_volume_scale_factor * importance_weight * (data_volume - self.min_dataset_len))
        # 考虑训练进度，越接近结束，可能希望k值越大
        progress_factor = (1 + (current_epoch / self.n_rounds))

        # 考虑损失变化率，损失变化小于某个阈值时，降低k值
        if avg_loss_change < 0.00015:
            loss_factor = 1.1
        else:
            loss_factor = 1

        # 带宽影响上界
        adjusted_k_upper_bound = self.k_max - math.ceil(
            self.bandwidth_scale_factor * (1 - importance_weight) * (self.bandwidth_max - bandwidth))

        # 确保上下界合理
        adjusted_k_lower_bound = max(self.k_min,
                                     min(self.k_max, math.ceil(adjusted_k_lower_bound * progress_factor * loss_factor)))
        adjusted_k_upper_bound = max(self.k_min, min(self.k_max, adjusted_k_upper_bound))

        return [
            k >= adjusted_k_lower_bound,
            k <= adjusted_k_upper_bound
        ]

    @staticmethod
    def compression_rate(k):
        return 0.00395 * k + 0.07567  # 近似线性函数

    def determine_k(self, current_epoch, avg_loss_change=1.0):
        # 优化变量
        k = cp.Variable((len(self.clients), len(self.clients[0].layer_importance_weights)), integer=True)
        # 目标函数：最小化通信成本
        objective_terms = []
        for i, client in enumerate(self.clients):
            # 对于每个客户端，计算每一层的目标函数值
            for j in range(len(client.layer_importance_weights)):
                objective_terms.append((self.compression_rate(k[i, j]) * self.weight_layer_sizes[j]
                                        + self.bias_layer_sizes[j]) / self.bandwidths[i])
        objective = cp.Minimize(cp.sum(objective_terms))

        # 约束条件列表 最大最小上下边界
        constraints = []
        for i, client in enumerate(self.clients):
            for j in range(len(client.layer_importance_weights)):
                constraints += self.interlayers_k_constraints(k[i, j],
                                                              current_epoch,
                                                              self.datasets_len[client.id],
                                                              self.bandwidths[client.id],
                                                              client.layer_importance_weights[j],
                                                              avg_loss_change)

        # 定义和求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)

        if problem.status in ["infeasible", "unbounded"]:
            print("Infeasible")
            return self.last_k_value
        else:
            self.last_k_value = k.value
            return k.value

    def calculate_k(self):
        alpha = 0.5
        print("Calculating k...")
        if self.current_rounds == 0:
            k_lists = self.determine_k(self.current_rounds)
        else:
            self.avg_loss_change = alpha * (abs(self.current_loss - self.last_loss)) + (
                    1 - alpha) * self.avg_loss_change  # EMA
            k_lists = self.determine_k(self.current_rounds, self.avg_loss_change)
        print(k_lists)
        for idx, k_list in enumerate(k_lists):
            self.clients[idx].assign_num_centroids(k_list)
        self.last_loss = self.current_loss

    def _init_clients(self):
        super()._init_clients()
        self.calculate_k()

    def train(self):
        self.calculate_k()
        super().train()
        self.current_rounds += 1

    def evaluate(self):
        result = super().evaluate()
        self.current_loss = result['loss']
        return result
