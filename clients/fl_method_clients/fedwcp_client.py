from copy import deepcopy
import torch
from torch.sparse import to_sparse_semi_structured
from clients.client import Client
from utils.kmeans import TorchKMeans


class FedWCPClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.reg_lambda = kwargs.get('reg_lambda', 0.01)
        self.n_clusters = kwargs.get('n_clusters', 16)
        self.base_decay_rate = hyperparam['base_decay_rate']
        self.global_model = self.preclustered_model_state_dict = self.new_clustered_model_state_dict = self.mask = None

    def receive_model(self, global_model):
        self.global_model = deepcopy(global_model).to(device=self.device)

    def init_client(self):
        self.model = deepcopy(self.global_model).to(device=self.device)
        self.preclustered_model_state_dict = self.model.state_dict()
        self.new_clustered_model_state_dict, self.mask = self._cluster_and_prune_model_weights()
        super().init_client()

    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}

        for key, weight in self.model.state_dict().items():
            if 'weight' in key and 'bn' not in key and 'downsample' not in key:
                original_shape = weight.shape
                # 判断是否是2D张量
                is_2d = (weight.dim() == 2)

                # 1) 若是2D => enforce_2of4=True，可以使用半结构化稀疏
                #    若不是2D => enforce_2of4=False，仅做KMeans稀疏(有0向量)但不强制2:4
                kmeans = TorchKMeans(
                    n_clusters=self.n_clusters,
                    is_sparse=True,
                    enforce_2of4=is_2d  # 仅2D时启用2:4
                )

                flattened_weights = weight.detach().view(-1, 1)
                kmeans.fit(flattened_weights)
                new_weights = kmeans.centroids[kmeans.labels_].view(original_shape)

                # 构造mask：非零位置为 True
                is_zero_centroid = (kmeans.centroids == 0).view(-1)
                mask = is_zero_centroid[kmeans.labels_].view(original_shape) == 0

                # 如果设备是CUDA && 是2D权重，才尝试转成半结构化稀疏
                if self.device.type == 'cuda' and is_2d:
                    # PyTorch半结构化稀疏主要在FP16生效
                    new_weights = new_weights.half()
                    new_weights = new_weights.masked_fill(~mask, 0)
                    new_weights = to_sparse_semi_structured(new_weights)

                mask_dict[key] = mask.bool()
                clustered_state_dict[key] = new_weights

            else:
                # 不剪枝的层，保持原状
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)

        return clustered_state_dict, mask_dict

    def _compute_global_local_model_difference(self):
        global_dict = self.global_model.state_dict()
        local_dict = self.preclustered_model_state_dict
        difference_dict = {}
        for key in global_dict:
            difference_dict[key] = local_dict[key] - global_dict[key]
        return difference_dict

    def _prune_model_weights(self):
        pruned_state_dict = {}
        for key, weight in self.model.state_dict().items():
            if key in self.mask:
                if weight.is_sparse:
                    dense_w = weight.to_dense()
                    pruned_w = dense_w * self.mask[key]
                    pruned_state_dict[key] = to_sparse_semi_structured(pruned_w.half())
                else:
                    pruned_state_dict[key] = weight * self.mask[key]
            else:
                pruned_state_dict[key] = weight
        return pruned_state_dict

    def train(self):
        ref_momentum = self._compute_global_local_model_difference()

        self.model.train()

        exponential_average_loss = None
        alpha = 0.5  # 损失平衡系数
        for epoch in range(self.epochs):
            for idx, (x, labels) in enumerate(self.client_train_loader):
                self.model.load_state_dict(self._prune_model_weights())

                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss_vec = self.criterion(outputs, labels)
                loss = loss_vec.mean()
                loss.backward()

                if exponential_average_loss is None:
                    exponential_average_loss = loss.item()
                else:
                    exponential_average_loss = alpha * loss.item() + (1 - alpha) * exponential_average_loss

                # 动量退火策略
                if loss.item() < exponential_average_loss:
                    decay_factor = min(self.base_decay_rate ** (idx + 1) * 1.1, 0.8)
                else:
                    decay_factor = max(self.base_decay_rate ** (idx + 1) / 1.1, 0.1)

                for name, param in self.model.named_parameters():
                    if name in ref_momentum:
                        param.grad += decay_factor * ref_momentum[name]

                self.optimizer.step()

        self.preclustered_model_state_dict = self.model.state_dict()
        self.new_clustered_model_state_dict, self.mask = self._cluster_and_prune_model_weights()
        return self.new_clustered_model_state_dict
