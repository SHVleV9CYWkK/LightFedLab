import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from servers.fl_method_servers.fedwcp_server import FedWCPServer


class AdFedWCPServer(FedWCPServer):
    def __init__(self, clients, model, device, args):
        self.n_rounds = args['n_rounds']
        self.k_round = args['k_round']
        self.current_rounds = 0
        self.avg_loss_change = float('inf')
        self.last_acc = self.current_acc = 0
        self.last_k_value = None
        self.k_min = 5
        self.k_max = 32
        self.datasets_len = {}
        self.max_dataset_len = 0
        self.min_dataset_len = float('inf')
        self.c_1 = self.c_2 = 32
        self.beta = args['beta']
        self.zeta = args['zeta']

        self.params_per_layer = {'weight': [], 'bias': []}
        for name, module in model.named_modules():
            if 'downsample' in name:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        self.params_per_layer['weight'].append(param.numel())
                    elif 'bias' in param_name:
                        self.params_per_layer['bias'].append(param.numel())

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

    def interlayers_k_constraints(self, current_epoch, data_volume, bandwidth, importance_weight,
                                  delta_accuracy=1.0):
        # 数据量影响下界
        data_volume_k_lower_bound = self.k_min + math.ceil(
            self.data_volume_scale_factor * importance_weight * (data_volume - self.min_dataset_len))
        # 考虑训练进度，越接近结束，可能希望k值越大
        progress_factor = (1 + (current_epoch / self.n_rounds))

        # 考虑损失变化率，损失变化小于某个阈值时，降低k值
        if delta_accuracy >= 1.0:
            accuracy_factor = 1 - self.beta * delta_accuracy
        else:
            accuracy_factor = 1 + self.zeta * abs(delta_accuracy)

        # 带宽影响上界
        adjusted_k_upper_bound = self.k_max - math.ceil(
            self.bandwidth_scale_factor * (1 - importance_weight) * (self.bandwidth_max - bandwidth))

        # 确保上下界合理
        adjusted_k_lower_bound = max(self.k_min, min(self.k_max, math.ceil(data_volume_k_lower_bound
                                                                           * progress_factor * accuracy_factor)))
        adjusted_k_upper_bound = max(self.k_min, min(self.k_max, adjusted_k_upper_bound))

        return adjusted_k_lower_bound, adjusted_k_upper_bound

    def objective_term_func(self, k, i, j):
        num_weights = self.params_per_layer['weight'][j]
        if j < len(self.params_per_layer['bias']):
            num_bias = self.params_per_layer['bias'][j]
        else:
            num_bias = 0
        return (self.c_1 * k + num_weights * torch.log2(k) + num_bias * self.c_2) / self.bandwidths[i]

    def determine_k(self, current_epoch, delta_accuracy=1.0):
        k = torch.nn.Parameter(torch.randint(self.k_min, self.k_max, (len(self.clients),
                                                                      len(self.clients[0].layer_importance_weights)),
                                             dtype=torch.float32,
                                             device=self.device if self.device.type == "cuda" or
                                                                   self.device.type == "cpu" else "cpu"))
        optimizer = optim.Adam([k], lr=0.05)

        for _ in range(self.k_round):
            optimizer.zero_grad()

            objective_terms = []
            for i, client in enumerate(self.clients):
                for j in range(len(client.layer_importance_weights)):
                    objective_terms.append(self.objective_term_func(k, i, j))

            loss = torch.sum(torch.stack(objective_terms))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for i, client in enumerate(self.clients):
                    for j in range(len(client.layer_importance_weights)):
                        lower_bound, upper_bound = self.interlayers_k_constraints(current_epoch,
                                                                                  self.datasets_len[client.id],
                                                                                  self.bandwidths[client.id],
                                                                                  client.layer_importance_weights[j],
                                                                                  delta_accuracy)
                        k[i, j].clamp_(lower_bound, upper_bound)

        return k.detach().round().int().cpu().numpy()

    def calculate_k(self):
        print("Calculating k...")
        start_time = time.time()
        if self.current_rounds == 0:
            k_lists = self.determine_k(self.current_rounds)
        else:
            k_lists = self.determine_k(self.current_rounds, self.current_acc - self.last_acc)
        print(k_lists)
        print(f"Calculate layer importance time: {time.time() - start_time}s")
        for idx, k_list in enumerate(k_lists):
            self.clients[idx].assign_num_centroids(k_list)
        self.last_acc = self.current_acc

    def compute_client_layer_weights(self):
        for client in self.selected_clients:
            client.compute_layer_weights()

    def train(self):
        self.calculate_k()
        super().train()
        self.compute_client_layer_weights()
        self.current_rounds += 1

    def evaluate(self):
        result, client_results = super().evaluate()
        self.current_acc = result['accuracy']
        return result, client_results
