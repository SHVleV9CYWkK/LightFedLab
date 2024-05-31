from abc import ABC, abstractmethod
import torch
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from copy import deepcopy


class Client(ABC):
    def __init__(self, dataset_index, full_dataset, bz, lr, epochs, criterion):
        self.model = None
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.train_dataset_len = len(dataset_index['train'])
        self.val_dataset_len = len(dataset_index['val'])
        self.num_classes = len(full_dataset.classes)
        client_train_dataset = Subset(full_dataset, indices=dataset_index['train'])
        client_val_dataset = Subset(full_dataset, indices=dataset_index['val'])
        self.client_train_loader = DataLoader(client_train_dataset, batch_size=bz, shuffle=False)
        self.client_val_loader = DataLoader(client_val_dataset, batch_size=bz, shuffle=False)


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def receive_model(self, global_model):
        pass

    def evaluate_local_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / self.val_dataset_len
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels,
                                               num_classes=len(self.client_val_loader.dataset.dataset.classes))
        precision = metrics.multiclass_precision(all_predictions, all_labels,
                                                 num_classes=len(self.client_val_loader.dataset.dataset.classes))
        recall = metrics.multiclass_recall(all_predictions, all_labels,
                                           num_classes=len(self.client_val_loader.dataset.dataset.classes))
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels,
                                         num_classes=len(self.client_val_loader.dataset.dataset.classes))

        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }


class FedAvgClient(Client):
    def __init__(self, dataset_index, fill_dataset, bz, lr, epochs, criterion, is_send_gradients=False):
        super().__init__(dataset_index, fill_dataset, bz, lr, epochs, criterion)
        self.is_send_gradients = is_send_gradients

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        initial_params = None
        if self.is_send_gradients:
            # 保存初始模型参数的拷贝
            initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
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

    def receive_model(self, global_model):
        self.model = deepcopy(global_model)