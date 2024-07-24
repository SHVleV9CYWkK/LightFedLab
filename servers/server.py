import random
import torch
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.utils import get_optimizer


class Server(ABC):
    def __init__(self, clients, model, device, optimizer_name, client_selection_rate=1, server_lr=0.01, n_job=1):
        self.clients = clients
        self.server_lr = server_lr
        self.device = device
        self.n_job = n_job
        self.client_selection_rate = client_selection_rate
        self.is_all_clients = client_selection_rate == 1
        if self.is_all_clients:
            self.selected_clients = clients
        else:
            self.selected_clients = random.sample(self.clients, len(clients) * self.client_selection_rate)
        self.model = model
        self.datasets_len = [client.train_dataset_len for client in self.clients]
        self._distribute_model()
        self._init_clients()
        self.optimizer_name = optimizer_name
        mp.set_start_method('spawn')

    @abstractmethod
    def _average_aggregate(self, weights_list):
        pass

    def _distribute_model(self):
        for client in self.clients:
            client.receive_model(self.model)

    def _init_clients(self):
        print("Initializing clients...")
        for client in self.clients:
            client.init_client()

    def _evaluate_model(self):
        result_list = []
        for client in self.selected_clients:
            result = client.evaluate_model()
            result_list.append(result)

        metrics_keys = result_list[0].keys()

        average_results = {key: 0 for key in metrics_keys}

        for result in result_list:
            for key in metrics_keys:
                average_results[key] += result.get(key, 0)

        for key in average_results.keys():
            average_results[key] /= len(result_list)

        return average_results

    def _handle_gradients(self, grad):
        return grad

    def _weight_aggregation(self, weights_list):
        datasets_len = self.datasets_len if self.is_all_clients else [client.dataset_len for client in
                                                                      self.selected_clients]
        total_len = sum(datasets_len)
        average_weights = {}
        for key in weights_list[0].keys():
            weighted_sum = sum(weights_list[client_id][key] * len_ for client_id, len_ in zip(weights_list, datasets_len))
            average_weights[key] = weighted_sum / total_len

        self.model.load_state_dict(average_weights)

    def _gradient_aggregation(self, weights_list, dataset_len=None):
        # 获取模型参数并确定设备
        global_weights = self.model.state_dict()

        # 准备用于累加梯度的字典，并确保所有张量都在同一设备上
        sum_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in global_weights.items()}

        # 如果未提供数据集长度，假设每个权重相同
        if dataset_len is None:
            dataset_len = torch.ones(len(weights_list), device=self.device)

        total_weight = sum(dataset_len)

        # 累加梯度
        for client_id, weight in zip(weights_list, dataset_len):
            for name, grad in weights_list[client_id].items():
                if grad is not None:
                    sum_gradients[name] += self._handle_gradients(grad).to(self.device) * weight  # 确保梯度在正确的设备上

        # 计算梯度的加权或非加权均值
        averaged_gradients = {name: sum_grad / total_weight for name, sum_grad in sum_gradients.items()}

        # # 检查梯度是否包含NaN或Inf
        for name, sum_grad in averaged_gradients.items():
            if torch.isnan(sum_grad).any() or torch.isinf(sum_grad).any():
                print(f"Gradient for {name} contains NaN or Inf.")

        optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.server_lr)
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

    @staticmethod
    def _execute_train_client(client):
        client_weights = client.train()
        return client.id, client_weights

    def _clients_train(self):
        self._sample_clients()
        if (self.device == 'cuda' or self.device == 'cpu') and self.n_job > 1:
            # 创建一个进程池，使用两个进程
            with Pool(2) as pool:
                # 使用map函数提交任务，它会自动分配任务给两个进程
                results = list(tqdm(pool.imap(self._execute_train_client, self.selected_clients), total=len(self.selected_clients)))
            # 将结果转换为字典
            locals_weights = {client_id: weights for client_id, weights in results}
        else:
            pbar = tqdm(total=len(self.selected_clients))
            locals_weights = dict()
            for client in self.selected_clients:
                client_weights = client.train()
                locals_weights[client.id] = client_weights
                pbar.update(1)
            pbar.clear()
            pbar.close()
        return locals_weights

    def train(self):
        clients_weights = self._clients_train()
        print("Aggregating models...")
        self._average_aggregate(clients_weights)
        self._distribute_model()

    def evaluate(self):
        print("Evaluating model...")
        average_eval_results = self._evaluate_model()
        return average_eval_results

    def lr_scheduler(self, metric):
        for client in self.selected_clients:
            client.update_lr(metric)
