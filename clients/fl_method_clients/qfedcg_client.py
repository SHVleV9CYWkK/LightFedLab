import torch
from torch.quantization import QuantStub, DeQuantStub, QConfig, default_observer
from clients.fl_method_clients.fedcg_client import FedCGClient


class QFedCGClient(FedCGClient):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, **kwargs)
        self.quantization_levels = kwargs.get('quantization_levels', 8)
        self.last_gradient = None
        self.qconfig = self.quantizer = self.dequantizer = None

    def initialize_quantization(self):
        custom_observer = default_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine,
                                                     quant_min=-2 ** (self.quantization_levels - 1),
                                                     quant_max=2 ** (self.quantization_levels - 1) - 1)
        self.qconfig = QConfig(activation=custom_observer, weight=custom_observer)
        self.quantizer = QuantStub()
        self.dequantizer = DeQuantStub()
        self.quantizer.qconfig = self.qconfig
        torch.quantization.prepare(self.quantizer, inplace=True)
        torch.quantization.convert(self.quantizer, inplace=True)

    def get_top_k_gradients(self, k):
        top_k_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                abs_grad = param.grad.abs().flatten()
                top_values, top_indices = torch.topk(abs_grad, k, largest=True, sorted=False)
                mask = torch.zeros_like(abs_grad, dtype=torch.bool)
                mask[top_indices] = True
                mask = mask.reshape(param.grad.shape)
                top_k_grads[name] = param.grad * mask
        return top_k_grads

    def quantize(self, tensor):
        quantized_tensor = self.quantizer(tensor)
        dequantized_tensor = self.dequantizer(quantized_tensor)
        return dequantized_tensor

    def compress_and_quantize_gradients(self):
        k = int(sum(p.numel() for p in self.model.parameters()) * self.compression_ratio)
        sparse_gradients = self.get_top_k_gradients(k)
        quantized_gradients = {}
        for name, grad in sparse_gradients.items():
            quantized_gradients[name] = self.quantize(grad)
        return quantized_gradients

    def calculate_quantization_levels(self, current_gradient):
        # 假设 self.last_gradient 存储了上一次的梯度
        if self.last_gradient is None:
            self.last_gradient = current_gradient
            return  # 第一次迭代时，没有之前的梯度可以比较

        # 计算梯度创新：当前梯度与上一次梯度的差的L2范数
        gradient_innovation = {name: (current_gradient[name] - self.last_gradient[name]).norm(2)
                               for name in current_gradient}

        # 基于梯度创新确定量化级别
        for name, innovation in gradient_innovation.items():
            if innovation > 1.0:  # 举例，阈值设置为1.0，可以根据实际情况调整
                self.quantization_levels = max(self.quantization_levels - 1, 1)
            else:
                self.quantization_levels = min(self.quantization_levels + 1, 8)

        # 更新 last_gradient 为当前梯度
        self.last_gradient = current_gradient

    def train(self):
        self.initialize_quantization()
        result = super().train()
        self.calculate_quantization_levels(result)
        return result
