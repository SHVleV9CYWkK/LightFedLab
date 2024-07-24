from abc import ABC, abstractmethod
import numpy as np
import torch
import torcheval.metrics.functional as metrics
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from utils.utils import get_optimizer, get_lr_scheduler


class Client(ABC):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, dl_n_job=0):
        self.id = client_id
        self.model = None
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.optimizer_name = hyperparam['optimizer_name']
        self.optimizer = None
        self.lr = hyperparam['lr']
        self.epochs = hyperparam['local_epochs']
        self.scheduler_name = hyperparam['scheduler_name']
        self.n_rounds = hyperparam['n_rounds']
        self.device = device
        train_indices = np.load(dataset_index['train']).tolist()
        val_indices = np.load(dataset_index['val']).tolist()
        self.train_dataset_len = len(train_indices)
        self.val_dataset_len = len(val_indices)
        self.num_classes = len(full_dataset.classes)
        client_train_dataset = Subset(full_dataset, indices=train_indices)
        client_val_dataset = Subset(full_dataset, indices=val_indices)
        self.client_train_loader = DataLoader(client_train_dataset, batch_size=hyperparam['bz'], num_workers=dl_n_job,
                                              shuffle=False, drop_last=True)
        self.client_val_loader = DataLoader(client_val_dataset, batch_size=hyperparam['bz'],
                                            shuffle=False, drop_last=True)
        self.global_metric = self.global_epoch = 0
        self.lr_scheduler = None

    @abstractmethod
    def train(self):
        pass

    def _local_train(self):
        self.model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

    def receive_model(self, global_model):
        if self.model is None:
            self.model = deepcopy(global_model).to(device=self.device)
        else:
            self.model.load_state_dict(global_model.state_dict())

    def init_client(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), self.lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    def update_lr(self, global_metric):
        self.lr_scheduler.step(global_metric)

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x).to(self.device)
                loss = self.criterion(outputs, labels)
                loss_meta_model = loss.mean()
                total_loss += loss_meta_model
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / len(self.client_val_loader)
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        # precision = metrics.multiclass_precision(all_predictions, all_labels, num_classes=self.num_classes)
        # recall = metrics.multiclass_recall(all_predictions, all_labels, num_classes=self.num_classes)
        # f1 = metrics.multiclass_f1_score(all_predictions, all_labels, num_classes=self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            # 'precision': precision.item(),
            # 'recall': recall.item(),
            # 'f1': f1.item()
        }
