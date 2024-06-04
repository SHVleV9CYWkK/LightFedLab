import time
import random
import torch
from torch.quantization import QuantStub, DeQuantStub, default_qconfig
from tqdm import tqdm
from abc import ABC, abstractmethod


class Server(ABC):
    def __init__(self, clients, model, client_selection_rate=1, server_lr=0.01):
        self.clients = clients
        self.selected_clients = None
        self.server_lr = server_lr
        self.client_selection_rate = client_selection_rate
        self.is_all_clients = client_selection_rate == 1
        self.model = model
        self.datasets_len = [client.train_dataset_len for client in self.clients]
        self._distribute_model()

    @abstractmethod
    def _average_aggregate(self):
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

    def _weight_aggregation(self):
        weights_list = [client.model.state_dict()
                        for client in (self.clients if self.is_all_clients else self.selected_clients)]
        datasets_len = self.datasets_len if self.is_all_clients else [client.dataset_len for client in
                                                                      self.selected_clients]
        average_weights = {}
        for key in weights_list[0].keys():
            weighted_sum = sum(weights[key] * len_ for weights, len_ in zip(weights_list, datasets_len))
            total_len = sum(datasets_len)
            average_weights[key] = weighted_sum / total_len

        self.model.load_state_dict(average_weights)

    def _handle_gradients(self, grad, client_id):
        return grad

    def _gradient_aggregation(self):
        gradient_sums = {}
        total_data_points = sum([client.train_dataset_len for client in self.selected_clients])

        for client in self.selected_clients:
            compressed_gradients = client.train()
            for key, compressed_grad in compressed_gradients.items():
                if key not in gradient_sums:
                    gradient_sums[key] = torch.zeros_like(compressed_grad)
                compressed_grad = self._handle_gradients(compressed_grad, client.id)
                # Decompress here if necessary, otherwise just add
                gradient_sums[key] += compressed_grad * (client.train_dataset_len / total_data_points)

        # Apply the aggregated gradients to the server model
        for name, param in self.model.named_parameters():
            param.grad = gradient_sums.get(name, torch.zeros_like(param))

        # Use an optimizer step to update model weights
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.server_lr)  # Assuming a learning rate
        optimizer.step()

    def _sample_clients(self):
        if self.client_selection_rate != 1:
            self.selected_clients = random.sample(self.clients, int(len(self.clients) * self.client_selection_rate))
        else:
            self.selected_clients = self.clients

    def train(self):
        self._sample_clients()
        pbar = tqdm(total=len(self.selected_clients))
        for client in self.selected_clients:
            client.train()
            pbar.update(1)
        pbar.clear()
        pbar.close()
        print("Aggregating models")
        start_time = time.time()
        self._average_aggregate()
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

    def _average_aggregate(self):
        is_send_gradients = self.clients[0].is_send_gradients
        self._weight_aggregation() if is_send_gradients else self._gradient_aggregation()


class FedCGServer(Server):
    def __init__(self, clients, model, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, client_selection_rate, server_lr)

    def _average_aggregate(self):
        self._gradient_aggregation()


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
        T = 1  # 假设每轮中每个客户端贡献一次更新，可以根据实际情况调整
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
    def create_server(self, fl_type, clients, model, client_selection_rate=1):
        if fl_type == 'fedavg':
            server_prototype = FedAvgServer
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, client_selection_rate)
