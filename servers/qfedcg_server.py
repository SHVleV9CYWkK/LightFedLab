import torch
from torch.quantization import QuantStub, DeQuantStub, default_qconfig
from servers.fedcd_server import FedCGServer


class QFedCGServer(FedCGServer):
    def __init__(self, clients, model, device, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, device, client_selection_rate, server_lr)
        self.quantization_levels = {client.id: 1 for client in clients}
        self.last_gradients = {client.id: None for client in clients}
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

    def train(self):
        [client.initialize_quantization() for client in self.clients]
        super().train()

    def _handle_gradients(self, quantized_tensor):
        current_gradient = self.dequantizer(quantized_tensor)
        return current_gradient