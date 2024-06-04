import time
import random
import torch
from torch.quantization import QuantStub, DeQuantStub, default_qconfig
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
        # 获取模型参数
        global_weights = self.model.state_dict()

        # 准备用于累加梯度的字典
        sum_gradients = {name: torch.zeros_like(param) for name, param in global_weights.items()}

        # 累加梯度
        if dataset_len is None:
            dataset_len = [1] * len(weights_list)  # 如果没有提供权重，使用等权重

        total_weight = sum(dataset_len)

        for gradients, weight in zip(weights_list, dataset_len):
            for name, grad in gradients.items():
                if grad is not None:
                    sum_gradients[name] += grad * weight

        # 计算梯度的加权或非加权均值
        averaged_gradients = {name: sum_grad / total_weight for name, sum_grad in sum_gradients.items()}

        for name, sum_grad in  averaged_gradients.items():
            if torch.isnan(sum_grad).any() or torch.isinf(sum_grad).any():
                print(f"Gradient for {name} contains NaN or Inf.")

        # 使用优化器更新模型参数
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.server_lr)
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


class FedAvgServer(Server):
    def __init__(self, clients, model, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, client_selection_rate, server_lr)

    def _average_aggregate(self, weights_list):
        is_send_gradients = self.clients[0].is_send_gradients
        self._weight_aggregation(weights_list) if is_send_gradients else self._gradient_aggregation(weights_list)


class FedCGServer(Server):
    def __init__(self, clients, model, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, client_selection_rate, server_lr)

    def _average_aggregate(self, weights_list):
        self._gradient_aggregation(weights_list)


class QFedCGServer(FedCGServer):
    def __init__(self, clients, model, client_selection_rate=1):
        super().__init__(clients, model, client_selection_rate)
        self.quantization_levels = {client.id: 1 for client in clients}
        self.last_gradients = {client.id: None for client in clients}
        self.quantization_errors = {client.id: 0 for client in clients}  # 初始化每个客户端的量化误差
        self.l_max = 8
        self.model_updates = []  # 存储模型更新的历史信息

        # 初始化量化和逆量化模块
        self.quantizer = QuantStub()
        self.dequantizer = DeQuantStub()

        # 设置量化配置
        self.quantizer.qconfig = default_qconfig
        self.dequantizer.qconfig = default_qconfig

        # 准备并转换量化器和逆量化器
        torch.quantization.prepare(self.quantizer, inplace=True)
        torch.quantization.convert(self.quantizer, inplace=True)
        torch.quantization.prepare(self.dequantizer, inplace=True)
        torch.quantization.convert(self.dequantizer, inplace=True)

    def _distribute_model(self):
        for client in self.clients if self.is_all_clients else self.selected_clients:
            client.receive_model(self.model)
            client.initialize_quantization()

    def calculate_psi_k(self):
        M = len(self.clients)  # 总客户端数量
        T = 1  # 每轮中每个客户端贡献一次更新
        I = len(self.model_updates)  # 模型更新历史的长度
        sum_model_updates = sum(self.model_updates[i] ** 2 for i in range(I))
        eta_k = 0.01  # 学习率，需要根据你的优化器配置设置

        psi_k = (1 / (eta_k ** 2 * M * T ** 2)) * sum_model_updates
        return psi_k

    def calculate_quantization_levels(self, client_id, current_gradient):
        last_gradient = self.last_gradients[client_id]
        if last_gradient is None:
            return 1  # 默认级别，适用于首次上传

        # 计算创新（梯度差异的二范数）
        innovation = torch.norm(current_gradient - last_gradient)

        # 计算阈值
        psi_k = self.calculate_psi_k()
        l_max = self.l_max
        l_prev = self.quantization_levels[client_id]
        quantization_error = 0.01  # 一个示例值，需要根据具体情况进行调整

        # 量化级别调整逻辑
        if innovation ** 2 >= psi_k + 3 * quantization_error * (l_max - l_prev + 1)**2:
            return min(l_prev + 1, l_max)
        return max(l_prev - 1, 1)

    def update_client_quant_config(self, client_id, quant_level):
        self.quantization_levels[client_id] = quant_level

    def _handle_gradients(self, quantized_tensor, client_id):
        current_gradient = self.dequantizer(quantized_tensor)
        current_gradient += self.quantization_errors[client_id]

        quant_level = self.calculate_quantization_levels(client_id, current_gradient)
        self.update_client_quant_config(client_id, quant_level)

        new_quantized_grad = self.quantizer(current_gradient)
        quantization_error = current_gradient - new_quantized_grad
        self.quantization_errors[client_id] = quantization_error  # 更新量化误差存储

        self.last_gradients[client_id] = current_gradient  # 更新上一次梯度记录
        return current_gradient


class ServerFactory:
    def create_server(self, fl_type, clients, model, client_selection_rate=1, server_lr=1e-3):
        if fl_type == 'fedavg':
            server_prototype = FedAvgServer
        elif fl_type == 'fedcg' or fl_type == 'qfedcg':
            server_prototype = FedCGServer
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, client_selection_rate, server_lr)
