from copy import deepcopy
import torch
from clinets.client import Client
from utils.kmeans import TorchKMeans


class FedWCPClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        self.reg_lambda = kwargs.get('reg_lambda', 0.01)
        self.global_model = None

    def receive_model(self, global_model):
        self.global_model = deepcopy(global_model).to(device=self.device)

    def init_local_model(self, model):
        self.model = deepcopy(model).to(device=self.device)

    def cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}

        for key, weight in self.model.state_dict().items():
            if 'weight' in key:
                original_shape = weight.shape
                kmeans = TorchKMeans(is_sparse=True)
                flattened_weights = weight.detach().view(-1, 1)
                kmeans.fit(flattened_weights)
                clustered_state_dict[key] = kmeans.centroids[kmeans.labels_].view(original_shape)
                is_zero_centroid = (kmeans.centroids == 0).view(-1)
                mask = is_zero_centroid[kmeans.labels_].view(original_shape) == 0
                mask_dict[key] = mask.bool()
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)
        return clustered_state_dict, mask_dict

    def compute_model_difference(self, global_model_state_dict):
        global_dict = global_model_state_dict
        local_dict = self.model.state_dict()
        difference_dict = {}
        for key in global_dict:
            difference_dict[key] = local_dict[key] - global_dict[key]
        return difference_dict

    def compute_sparse_refined_regularization(self, mask):
        regularization_terms = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                regularization_terms[name] = self.reg_lambda * ~mask[name] * param.data
        return regularization_terms

    def prune_model_weights(self, mask_dict):
        pruned_state_dict = {}
        for key, weight in self.model.state_dict().items():
            if key in mask_dict:
                pruned_weight = weight * mask_dict[key]
                pruned_state_dict[key] = pruned_weight
            else:
                pruned_state_dict[key] = weight
        return pruned_state_dict

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        initial_global_params = {name: param.clone() for name, param in self.global_model.named_parameters()}
        clustered_model_state_dict, mask = self.cluster_and_prune_model_weights()
        self.model.load_state_dict(clustered_model_state_dict)
        momentum = self.compute_model_difference(initial_global_params)
        regularization_terms = self.compute_sparse_refined_regularization(mask)

        self.model.train()
        decay_rate = 0.9
        for epoch in range(self.epochs):
            for idx, (x, labels) in enumerate(self.client_train_loader):
                decay_factor = decay_rate ** idx
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                for name, param in self.model.named_parameters():
                    if name in momentum:
                        param.grad += decay_factor * momentum[name]
                        if 'weight' in name:
                            param.grad += regularization_terms[name]
                optimizer.step()
                pruned_model_state_dict = self.prune_model_weights(mask)
                self.model.load_state_dict(pruned_model_state_dict)
            return self.model.state_dict()
