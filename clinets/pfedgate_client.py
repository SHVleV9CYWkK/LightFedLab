import torch
from clinets.client import Client
from models.gating_layers import GatingLayer


class PFedGateClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        data_sample, _ = full_dataset[0]
        self.input_feat_size = data_sample.numel()
        self.num_channels = data_sample.size(0)
        self.gating_layer = None

    def init_gating_layer(self):
        self.gating_layer = GatingLayer(self.model, self.device, self.input_feat_size, self.num_channels)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.gating_layer.parameters()), lr=self.lr)
        initial_model_params = {name: param.clone() for name, param in self.model.named_parameters()}
        initial_gating_layer_params = {name: param.clone() for name, param in self.gating_layer.named_parameters()}

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                gating_weights = self.gating_layer(x)
                self._prune_model_weights(gating_weights)
                outputs = self.model(x) * gating_weights
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        total_model_gradients = {}
        for name, param in self.model.named_parameters():
            if initial_model_params[name] is not None:
                # 计算总梯度变化
                total_gradient_change = initial_model_params[name].data - param.data
                total_model_gradients[name] = total_gradient_change

        return total_model_gradients

    def _prune_model_weights(self, mask):
        for name, param in self.model.named_parameters():
            param.data = param.data * mask[name]