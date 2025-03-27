import torch
from clients.client import Client


class FedPACClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        # 从 hyperparam 中读取特征对齐正则系数
        self.lam_align = hyperparam.get('lam_align', 1.0)
        # 用于存储来自服务器的全局特征中心
        self.global_centroids = None

    def set_global_centroids(self, centroids):
        self.global_centroids = centroids.to(self.device)

    def train(self):
        if self.optimizer is None:
            self.init_client()

        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model.model, 'fc'):
            for param in self.model.model.fc.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError("There is no classifier")

        classifier_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=self.lr, momentum=0.9)

        self.model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer_classifier.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                optimizer_classifier.step()

        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model.model, 'fc'):
            # 不让 fc 更新
            for name, param in self.model.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = True
        else:
            raise NotImplementedError("There is no fc, can't freeze/unfreeze properly")

        # 创建只包含特征提取器参数的 optimizer
        feature_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_feature = torch.optim.SGD(feature_params, lr=self.lr, momentum=0.9)

        remain_epochs = max(0, self.epochs - 1)
        for _ in range(remain_epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer_feature.zero_grad()

                features = self.model.feature_extractor(x)
                outputs = self.model.classifier(features)
                ce_loss = self.criterion(outputs, labels).mean()

                align_loss = 0.0
                if self.global_centroids is not None:
                    batch_size = x.shape[0]
                    centroid_vectors = self.global_centroids[labels]
                    diff = features - centroid_vectors
                    align_loss = torch.sum(diff * diff) / (batch_size)

                loss = ce_loss + self.lam_align * align_loss
                loss.backward()
                optimizer_feature.step()

        return self.model.state_dict()
