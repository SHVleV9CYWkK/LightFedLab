import torch
from torch.quantization import QuantStub, DeQuantStub, QConfig, default_observer
from fedcg_client import FedCGClient


class QFedCGClient(FedCGClient):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs)
        self.quantization_levels = kwargs.get('quantization_levels', 8)
        self.initialize_quantization()

    def initialize_quantization(self):
        custom_observer = default_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine,
                                                     quant_min=-2**(self.quantization_levels-1),
                                                     quant_max=2**(self.quantization_levels-1) - 1)
        self.qconfig = QConfig(activation=custom_observer(), weight=custom_observer())
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

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                compressed_and_quantized_grads = self.compress_and_quantize_gradients()
                for name, param in self.model.named_parameters():
                    if name in compressed_and_quantized_grads:
                        param.grad = compressed_and_quantized_grads[name]
                optimizer.step()
        return self.model.state_dict()