import torch
from torch.quantization import QuantStub, DeQuantStub, default_qconfig
from fedcd_server import FedCGServer


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