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

    def compress_gradients(self):
        top_k = int(len(self.model.parameters()) * self.compression_ratio)
        # k 是你想保留的梯度元素的数量
        gradient_list = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 将梯度的绝对值和相应的名称、索引打包
                grads_flat = param.grad.view(-1)
                gradient_list.extend(zip(grads_flat.abs(), itertools.repeat(name), range(len(grads_flat))))

        # 排序找到最大的k个梯度
        top_k_gradients = sorted(gradient_list, key=lambda x: x[0], reverse=True)[:top_k]

        # 生成一个字典来保存压缩后的梯度
        compressed_gradients = {name: torch.zeros_like(param.grad) for name, param in self.model.named_parameters()}
        for _, name, idx in top_k_gradients:
            # 将Top-k梯度放回它们原来的位置
            idx_tuple = np.unravel_index(idx, compressed_gradients[name].shape)
            compressed_gradients[name][idx_tuple] = self.model.named_parameters()[name].grad[idx_tuple]

        return compressed_gradients

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

                # Apply gradient compression here based on self.compression_ratio
                # This could involve selecting the top-k gradients, for example
                self.compress_gradients()

                optimizer.step()

        # Assuming we only send compressed gradients
        return self.compress_gradients()


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
