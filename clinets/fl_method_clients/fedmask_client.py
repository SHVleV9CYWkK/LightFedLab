import torch
import torcheval.metrics.functional as metrics
from clinets.client import Client
from utils.utils import get_optimizer, get_lr_scheduler
from models.fedmask.mask_model import MaskedModel


class FedMaskClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device,
                 **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.mask_model = None
        self.pruning_rate = kwargs.get('sparse_factor', 0.2)

    def _local_train(self):
        self.mask_model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.mask_model(x)
                loss = self.criterion(outputs, labels).mean()
                print(f"Client{self.id}: Loss:{loss.item()}")
                loss.backward()
                self.optimizer.step()

    def receive_mask(self, masks):
        for name, param in self.mask_model.named_parameters():
            if name in masks:
                param.data = masks[name]

    def train(self):
        self._local_train()
        return self._binarize_mask()

    def _client_pruning(self):
        self._local_train()
        self._prune_mask()
        return self._binarize_mask()

    def _prune_mask(self):
        mask_parameters = {name: param for name, param in self.mask_model.named_parameters() if "mask_" in name}
        for name, mask in mask_parameters.items():
            k = int(self.pruning_rate * mask.numel())
            threshold = torch.topk(mask.view(-1), k, largest=True).values.min()
            mask.data[mask.data < threshold] = 0

    def _binarize_mask(self):
        # Binary mask using sigmoid and threshold of 0.5
        binary_mask = {}
        for name, param in self.mask_model.named_parameters():
            if "mask_" in name:
                binary_mask[name] = (torch.sigmoid(param) > 0.5).float()
        return binary_mask

    def init_client(self):
        self.mask_model = MaskedModel(self.model)
        self.optimizer = get_optimizer(self.optimizer_name, self.mask_model.parameters(), self.lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, 'reduce_on_plateau')
        return self._client_pruning()

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.mask_model(x).to(self.device)
                loss = self.criterion(outputs, labels).mean()
                total_loss += loss
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
