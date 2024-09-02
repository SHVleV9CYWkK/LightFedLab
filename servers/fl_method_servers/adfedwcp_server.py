import math
import random
import time
import torch
import torch.optim as optim
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
        mean_bandwidth = (self.bandwidth_max + self.bandwidth_min) / 2
        std_bandwidth = (self.bandwidth_max - self.bandwidth_min) / 6
        self.bandwidths = [max(self.bandwidth_min, min(self.bandwidth_max,
                                                       round(random.normalvariate(mean_bandwidth, std_bandwidth))))
                           for _ in range(len(clients))]

        print(f"The bandwidth of the clients is: {self.bandwidths}")

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

        return adjusted_k_lower_bound, adjusted_k_upper_bound

    @staticmethod
    def compression_rate(k):
        return 0.046985 * torch.log(k) + 0.008387   # 近似对数函数

    def determine_k(self, current_epoch, avg_loss_change=1.0):
        k = torch.nn.Parameter(torch.randint(self.k_min, self.k_max, (len(self.clients),
                                                                      len(self.clients[0].layer_importance_weights)),
                                             dtype=torch.float32,
                                             device=self.device if self.device.type == "cuda" or
                                                                   self.device.type == "cpu" else "cpu"))
        optimizer = optim.Adam([k], lr=0.05)

        for _ in range(1000):
            optimizer.zero_grad()

            objective_terms = []
            for i, client in enumerate(self.clients):
                for j in range(len(client.layer_importance_weights)):
                    objective_terms.append(
                        (self.compression_rate(k[i, j]) * self.weight_layer_sizes[j]
                         + self.bias_layer_sizes[j]) / self.bandwidths[i]
                    )

            loss = torch.sum(torch.stack(objective_terms))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for i, client in enumerate(self.clients):
                    for j in range(len(client.layer_importance_weights)):
                        lower_bound, upper_bound = self.interlayers_k_constraints(k[i, j],
                                                                                  current_epoch,
                                                                                  self.datasets_len[client.id],
                                                                                  self.bandwidths[client.id],
                                                                                  client.layer_importance_weights[j],
                                                                                  avg_loss_change)
                        k[i, j].clamp_(lower_bound, upper_bound)

        return k.detach().round().int().cpu().numpy()

    def calculate_k(self):
        alpha = 0.5
        print("Calculating k...")
        start_time = time.time()
        if self.current_rounds == 0:
            k_lists = self.determine_k(self.current_rounds)
        else:
            self.avg_loss_change = alpha * (abs(self.current_loss - self.last_loss)) + (
                    1 - alpha) * self.avg_loss_change  # EMA
            k_lists = self.determine_k(self.current_rounds, self.avg_loss_change)
        print(k_lists)
        print(f"Calculate layer importance time: {time.time() - start_time}s")
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
