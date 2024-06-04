import itertools
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.quantization import QuantStub, DeQuantStub, QConfig, default_observer
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from copy import deepcopy


class Client(ABC):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device):
        self.id = client_id
        self.model = None
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.device = device
        train_indices = np.load(dataset_index['train']).tolist()
        val_indices = np.load(dataset_index['val']).tolist()
        self.train_dataset_len = len(train_indices)
        self.val_dataset_len = len(val_indices)
        self.num_classes = len(full_dataset.classes)
        client_train_dataset = Subset(full_dataset, indices=train_indices)
        client_val_dataset = Subset(full_dataset, indices=val_indices)
        self.client_train_loader = DataLoader(client_train_dataset, batch_size=bz, shuffle=False)
        self.client_val_loader = DataLoader(client_val_dataset, batch_size=bz, shuffle=False)

    @abstractmethod
    def train(self):
        pass

    def receive_model(self, global_model):
        self.model = deepcopy(global_model).to(device=self.device)

    def evaluate_local_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x).to(self.device)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / self.val_dataset_len
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        precision = metrics.multiclass_precision(all_predictions, all_labels, num_classes=self.num_classes)
        recall = metrics.multiclass_recall(all_predictions, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, num_classes=self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }


class FedAvgClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        self.is_send_gradients = kwargs.get('is_send_gradients', False)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        initial_params = None
        if self.is_send_gradients:
            # 保存初始模型参数的拷贝
            initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        if not self.is_send_gradients:
            return self.model.state_dict()

        # 计算从初始到当前的总梯度变化
        total_gradients = {}
        for name, param in self.model.named_parameters():
            if initial_params[name] is not None:
                # 计算总梯度变化
                total_gradient_change = initial_params[name].data - param.data
                total_gradients[name] = total_gradient_change
        return total_gradients


class FedCGClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        self.compression_ratio = kwargs.get('compression_ratio', 1)

    def compress_gradients(self, param_changes):
        total_params = sum(p.numel() for p in param_changes.values())
        top_k = int(total_params * self.compression_ratio)

        # 使用一个列表来保存所有的参数变化量及其绝对值
        all_changes = []
        shapes = {}

        for name, change in param_changes.items():
            changes_flat = change.flatten()
            # 记录每个参数的形状以便重构
            shapes[name] = change.shape
            all_changes.append(changes_flat.abs())

        # 合并所有参数变化量为一个单一的向量
        all_changes = torch.cat(all_changes)
        # 找到变化量中最大的top_k个元素的位置
        _, top_k_indices = torch.topk(all_changes, top_k)

        # 生成一个字典来保存压缩后的参数变化量
        compressed_param_changes = {name: torch.zeros_like(change) for name, change in param_changes.items()}

        # 分配top-k变化量到对应的参数
        idx_offset = 0
        for name, change in param_changes.items():
            num_elements = np.prod(shapes[name])
            # 获取当前参数对应的top-k索引
            current_indices = top_k_indices[(top_k_indices >= idx_offset) & (top_k_indices < idx_offset + num_elements)] - idx_offset
            # 将选择的变化量放回其原位置
            compressed_param_changes[name].view(-1)[current_indices] = change.view(-1)[current_indices]
            idx_offset += num_elements

        return compressed_param_changes

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        param_changes = {name: initial_params[name] - param.data for name, param in self.model.named_parameters()}
        return self.compress_gradients(param_changes)


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


class ClientFactory:
    def create_client(self, num_client, fl_type, dataset_index, full_dataset,
                      bz, lr, epochs, criterion, device, **kwargs):

        if fl_type == 'fedavg':
            client_prototype = FedAvgClient
        elif fl_type == 'fedcg':
            client_prototype = FedCGClient
        elif fl_type == 'qfedcg':
            client_prototype = QFedCGClient
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')
        clients = []
        for idx in range(num_client):
            clients.append(client_prototype(idx,
                                            dataset_index[idx],
                                            full_dataset,
                                            bz,
                                            lr,
                                            epochs,
                                            criterion,
                                            device,
                                            **kwargs))

        return clients
