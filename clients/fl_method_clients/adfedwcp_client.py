import numpy as np
import torch
import torch.nn as nn
from clients.fl_method_clients.fedwcp_client import FedWCPClient
from models.adfedwcp.imprint_classifier import ImprintClassifier
from utils.kmeans import TorchKMeans


class AdFedWCPClient(FedWCPClient):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, **kwargs)
        self.num_centroids = {}
        self.layer_importance = None
        self.layer_statistics = {}

    @staticmethod
    def calculate_embedding_length(feature_map):
        c, h, w = feature_map.size(1), feature_map.size(2), feature_map.size(3)
        return min(100, c * h * w)

    @staticmethod
    def to_feature_map(output, batch_size):
        """把层输出整理为逐样本的 [B, C, H, W]：卷积输出不变，全连接的 [B, C] 视为 1×1，
        Transformer 的 token 输出 [B, N, C]（窗口注意力为 [B·窗口数, N, C]）排成 [B, C, 全部 token, 1]。"""
        if output.dim() == 4 and output.size(0) == batch_size:
            return output
        if output.dim() == 2:
            return output[:, :, None, None]
        tokens = output.reshape(batch_size, -1, output.size(-1))
        return tokens.transpose(1, 2).unsqueeze(-1)

    def _imprinted_layer_names(self):
        # 与服务器的层顺序一致：named_modules 中除 downsample 外的 Conv2d / Linear
        return [name for name, layer in self.model.named_modules()
                if 'downsample' not in name and isinstance(layer, (nn.Conv2d, nn.Linear))]

    def get_all_layer_outputs(self, x):
        # 按前向执行顺序记录各层输出，键为模块名
        outputs = {}
        modules = dict(self.model.named_modules())
        hooks = [modules[name].register_forward_hook(
                     lambda module, inputs, output, name=name: outputs.__setitem__(name, output.detach()))
                 for name in self._imprinted_layer_names()]
        with torch.no_grad():
            self.model(x)
        for hook in hooks:
            hook.remove()
        return outputs

    def compute_layer_importance(self):
        self.model.eval()
        layer_accuracies = {}
        layer_counts = {}

        for x, labels in self.client_train_loader:
            x, labels = x.to(self.device), labels.to(self.device)

            outputs = self.get_all_layer_outputs(x)

            for name, output in outputs.items():
                feature_map = self.to_feature_map(output, labels.size(0))
                imprint_clf = ImprintClassifier(feature_map.size(1), self.num_classes,
                                                self.calculate_embedding_length(feature_map), device=self.device)
                imprint_clf.imprint_weights(feature_map, labels)

                logits = imprint_clf(feature_map)
                _, predicted = torch.max(logits, 1)
                correct = (predicted == labels).float().sum()
                accuracy = correct / labels.size(0)

                if name in layer_accuracies:
                    layer_accuracies[name] += accuracy.item()
                    layer_counts[name] += 1
                else:
                    layer_accuracies[name] = accuracy.item()
                    layer_counts[name] = 1

        for name in layer_accuracies:
            layer_accuracies[name] /= layer_counts[name]

        # I_j = Acc_j - Acc_{j-1} 按前向执行顺序计算，第一层与随机猜测比较
        layer_importance = {}
        previous_accuracy = 1 / self.num_classes
        for name, accuracy in layer_accuracies.items():
            layer_importance[name] = accuracy - previous_accuracy
            previous_accuracy = accuracy

        # 按服务器的层顺序返回；没有输出的层记为 0
        return {name: layer_importance.get(name, 0.0) for name in self._imprinted_layer_names()}

    def compute_number_of_layers(self):
        return len(self._imprinted_layer_names())

    def compute_layer_weights(self, uniform=False):
        # 上传各层的原始重要性 I_j；带温度的 softmax 归一化 (式 4) 由服务器完成。
        # uniform 时，或本地数据不足一个完整 batch（drop_last 下没有数据可用于 imprinting）时，所有层同等重要
        layer_importance = {} if uniform else self.compute_layer_importance()
        if layer_importance:
            self.layer_importance = np.array(list(layer_importance.values()))
        else:
            self.layer_importance = np.zeros(self.compute_number_of_layers())

    def compute_layer_statistics(self):
        # 每个聚类层被保留（未剪枝）权重的方差 σ² 与数量 W，见设计文档 1.3 节
        self.layer_statistics = {}
        for key, weight in self.preclustered_model_state_dict.items():
            if self._should_compress(key, weight):
                retained = weight[self.mask[key]]
                variance = retained.var().item() if retained.numel() > 1 else 0.0
                self.layer_statistics[key] = (variance, retained.numel())

    def assign_num_centroids(self, num_centroids):
        self.num_centroids.update(num_centroids)

    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}
        for key, weight in self.model.state_dict().items():
            if self._should_compress(key, weight):
                original_shape = weight.shape
                kmeans = TorchKMeans(n_clusters=self.num_centroids[key], is_sparse=True)
                flattened_weights = weight.detach().view(-1, 1)
                kmeans.fit(flattened_weights)

                new_weights = kmeans.centroids[kmeans.labels_].view(original_shape)
                is_zero_centroid = (kmeans.centroids == 0).view(-1)
                mask = is_zero_centroid[kmeans.labels_].view(original_shape) == 0
                mask_dict[key] = mask.bool()
                clustered_state_dict[key] = new_weights
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)
        return clustered_state_dict, mask_dict

    def init_client(self):
        for key, weight in self.global_model.state_dict().items():
            if self._should_compress(key, weight):
                self.num_centroids[key] = 8
        super().init_client()
        self.compute_layer_weights()
        self.compute_layer_statistics()
        print(f"Client{self.id} initialized successfully")

    def train(self):
        result = super().train()
        return result
