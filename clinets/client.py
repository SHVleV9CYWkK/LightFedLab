from abc import ABC, abstractmethod
import numpy as np
import torch
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from fedavg_client import FedAvgClient
from fedcg_client import FedCGClient
from qfedcg_client import QFedCGClient


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
