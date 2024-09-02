import numpy as np
import torch
import torch.nn as nn
from clients.fl_method_clients.fedwcp_client import FedWCPClient
from models.adfedwcp.imprint_classifier import ImprintClassifier
from utils.kmeans import TorchKMeans


class AdFedWCPClient(FedWCPClient):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, **kwargs)
        self.layer_importance_weights = None
        self.num_centroids = {}
        self.layer_importance = {}

    @staticmethod
    def calculate_embedding_length(output):
        if output.dim() == 4:  # 对于卷积层输出
            c, h, w = output.size(1), output.size(2), output.size(3)
            return min(100, c * h * w)
        else:  # 对于全连接层输出
            return output.size(1)

    def get_all_layer_outputs(self, x):
        outputs = {}

        def hook(module, input, output):
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module_name = module.__class__.__name__
                if module_name in outputs:
                    module_name += f"_{len(outputs)}"
                outputs[module_name] = output.detach()

        hooks = []
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hooks.append(layer.register_forward_hook(hook))

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
                num_channels = output.size(1)
                embedding_length = self.calculate_embedding_length(output)
                imprint_clf = ImprintClassifier(num_channels, self.num_classes, embedding_length, device=self.device)
                if output.dim() == 4:
                    imprint_clf.imprint_weights(output, labels)  # 对卷积层输出应用池化并计算权重
                else:
                    imprint_clf.imprint_weights(output.view(output.size(0), -1, 1, 1), labels)  # 全连接层的处理

                logits = imprint_clf(output)
                _, predicted = torch.max(logits, 1)
                correct = (predicted == labels).float().sum()
                accuracy = correct / labels.size(0)

                if name in layer_accuracies:
                    layer_accuracies[name] += accuracy.item()
                    layer_counts[name] += 1
                else:
                    layer_accuracies[name] = accuracy.item()
                    layer_counts[name] = 1

        # 计算每层的平均准确率
        for name in layer_accuracies:
            layer_accuracies[name] /= layer_counts[name]

        layer_importance = {}
        previous_accuracy = 0
        for name, accuracy in layer_accuracies.items():
            layer_importance[name] = accuracy - previous_accuracy
            previous_accuracy = accuracy

        return layer_importance

    def compute_layer_weights(self):
        layer_importance = self.compute_layer_importance()

        importance_values = np.array(list(layer_importance.values()))
        exp_values = np.exp(importance_values)
        softmax_values = exp_values / np.sum(exp_values)

        self.layer_importance_weights = softmax_values

    def assign_num_centroids(self, k_list):
        index = 0
        for key, weight in self.global_model.state_dict().items():
            if 'weight' in key:
                self.num_centroids[key] = int(k_list[index])
                index += 1

    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}
        for key, weight in self.model.state_dict().items():
            if 'weight' in key:
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
            if 'weight' in key:
                self.num_centroids[key] = 8
        super().init_client()
        self.compute_layer_weights()

    def train(self):
        result = super().train()
        self.compute_layer_weights()
        return result
