import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprintClassifier(nn.Module):
    def __init__(self, num_channels, num_classes=10, embedding_length=100, device='cpu'):
        super(ImprintClassifier, self).__init__()
        target_size = round(math.sqrt(embedding_length / num_channels))
        self.target_size = target_size
        self.num_channels = num_channels
        input_features = target_size * target_size * num_channels
        self.fc = nn.Linear(input_features, num_classes)
        self.weights = None  # 存储预计算的权重
        self.device = device

    def imprint_weights(self, feature_maps, labels):
        pooled_features = F.adaptive_avg_pool2d(feature_maps, (self.target_size, self.target_size))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        self.weights = torch.zeros(self.fc.out_features, pooled_features.size(1), device=self.device)
        count = torch.zeros(self.fc.out_features)
        for i in range(labels.size(0)):
            self.weights[labels[i]] += pooled_features[i].to(self.device)
            count[labels[i]] += 1
        for i in range(self.fc.out_features):
            if count[i] > 0:
                self.weights[i] /= count[i]

    def forward(self, x):
        if self.weights is None:
            raise ValueError("Weights have not been imprinted.")
        if x.dim() == 4:  # For convolutional layers
            x = F.adaptive_avg_pool2d(x, (self.target_size, self.target_size))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return F.linear(x, self.weights)
