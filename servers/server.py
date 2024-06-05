import time
import random
import torch

from tqdm import tqdm
from abc import ABC, abstractmethod


class Server(ABC):
    def __init__(self, clients, model, client_selection_rate=1, server_lr=0.01):
        self.clients = clients
        self.server_lr = server_lr
        self.client_selection_rate = client_selection_rate
        self.is_all_clients = client_selection_rate == 1
        if self.is_all_clients:
            self.selected_clients = clients
        else:
            self.selected_clients = random.sample(self.clients, len(clients) * self.client_selection_rate)
        self.model = model
        self.datasets_len = [client.train_dataset_len for client in self.clients]
        self._distribute_model()

    @abstractmethod
    def _average_aggregate(self, weights_list):
        pass

    def _distribute_model(self):
        for client in self.clients if self.is_all_clients else self.selected_clients:
            client.receive_model(self.model)

    def _evaluate_model(self):
        result_list = []
        for client in self.selected_clients:
            result = client.evaluate_local_model()
            result_list.append(result)

        metrics_keys = result_list[0].keys()

        average_results = {key: 0 for key in metrics_keys}

        for result in result_list:
            for key in metrics_keys:
                average_results[key] += result.get(key, 0)

        for key in average_results.keys():
            average_results[key] /= len(result_list)

        return average_results

    def _handle_gradients(self, grad, client_id):
        return grad

    def _weight_aggregation(self, weights_list):
        datasets_len = self.datasets_len if self.is_all_clients else [client.dataset_len for client in
                                                                      self.selected_clients]
        average_weights = {}
        for key in weights_list[0].keys():
            weighted_sum = sum(weights[key] * len_ for weights, len_ in zip(weights_list, datasets_len))
            total_len = sum(datasets_len)
            average_weights[key] = weighted_sum / total_len

        self.model.load_state_dict(average_weights)

    def _gradient_aggregation(self, weights_list, dataset_len=None):
        # 获取模型参数并确定设备
        global_weights = self.model.state_dict()
        device = next(self.model.parameters()).device  # 从模型的第一个参数获取设备信息

        # 准备用于累加梯度的字典，并确保所有张量都在同一设备上
        sum_gradients = {name: torch.zeros_like(param).to(device) for name, param in global_weights.items()}

        # 如果未提供数据集长度，假设每个权重相同
        if dataset_len is None:
            dataset_len = torch.ones(len(weights_list), device=device)

        total_weight = sum(dataset_len)

        # 累加梯度
        for gradients, weight in zip(weights_list, dataset_len):
            for name, grad in gradients.items():
                if grad is not None:
                    sum_gradients[name] += grad.to(device) * weight  # 确保梯度在正确的设备上

        # 计算梯度的加权或非加权均值
        averaged_gradients = {name: sum_grad / total_weight for name, sum_grad in sum_gradients.items()}

        # 检查梯度是否包含NaN或Inf
        for name, sum_grad in averaged_gradients.items():
            if torch.isnan(sum_grad).any() or torch.isinf(sum_grad).any():
                print(f"Gradient for {name} contains NaN or Inf.")

        # 使用优化器更新模型参数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.server_lr)
        for name, param in self.model.named_parameters():
            if name in averaged_gradients:
                param.grad = averaged_gradients[name]

        optimizer.step()
        optimizer.zero_grad()

    def _sample_clients(self):
        if self.client_selection_rate != 1:
            self.selected_clients = random.sample(self.clients, int(len(self.clients) * self.client_selection_rate))
        else:
            self.selected_clients = self.clients

    def train(self):
        self._sample_clients()
        pbar = tqdm(total=len(self.selected_clients))
        local_weights = []
        for client in self.selected_clients:
            client_weights = client.train()
            local_weights.append(client_weights)
            pbar.update(1)
        pbar.clear()
        pbar.close()
        print("Aggregating models")
        start_time = time.time()
        self._average_aggregate(local_weights)
        end_time = time.time()
        print(f"Aggregation takes {(end_time - start_time):.3f} seconds")
        self._distribute_model()

    def evaluate(self):
        print("Evaluating model")
        average_eval_results = self._evaluate_model()
        return average_eval_results


class ServerFactory:
    def create_server(self, fl_type, clients, model, client_selection_rate=1, server_lr=1e-3):
        if fl_type == 'fedavg':
            server_prototype = FedAvgServer
        elif fl_type == 'fedcg' or fl_type == 'qfedcg':
            server_prototype = FedCGServer
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, client_selection_rate, server_lr)
