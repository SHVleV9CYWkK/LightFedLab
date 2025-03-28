import torch
import torch.nn as nn
import torch.nn.functional as F
import torcheval.metrics.functional as metrics
from clients.client import Client


class IntermediateFeatureHook:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output


class GaussianWrapper(nn.Module):

    def __init__(self, base_model, hooking_layer, embedding_dim=512, latent_dim=128, num_classes=10):
        super().__init__()
        self.base_model = base_model
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.hook = IntermediateFeatureHook()
        hooking_layer.register_forward_hook(self.hook.hook_fn)

        self.mean_layer = nn.Linear(embedding_dim, latent_dim)
        self.logvar_layer = nn.Linear(embedding_dim, latent_dim)

        self.classifier = nn.Linear(latent_dim, num_classes)

    def extract_feature(self, x):
        self.hook.features = None
        _ = self.base_model(x)
        feats = self.hook.features

        if len(feats.shape) == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)

        return feats

    def forward(self, x):
        feats = self.extract_feature(x)
        z_mean = self.mean_layer(feats)
        z_logvar = self.logvar_layer(feats)

        logits = self.classifier(z_mean)

        return z_mean, z_logvar, logits


class FedCRClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.beta = hyperparam.get("beta", 1e-3)
        self.global_dist = None

    def set_global_dist(self, global_dist):
        self.global_dist = global_dist

    def receive_model(self, global_model):
        if hasattr(global_model, 'model') and hasattr(global_model.model, 'avgpool'):
            hooking_layer = global_model.model.avgpool
            embedding_dim = 512
        else:
            raise NotImplementedError("请指定合适的hooking_layer")

        wrapped = GaussianWrapper(
            base_model=global_model,  # 注意这里直接传入 global_model
            hooking_layer=hooking_layer,
            embedding_dim=embedding_dim,
            latent_dim=128,  # 可从超参获取
            num_classes=self.num_classes
        )
        self.model = wrapped.to(self.device)

    def _kl_gaussian_diag(self, z_mean, z_logvar, g_mean, g_sigma):
        p_var = z_logvar.exp()
        q_var = g_sigma * g_sigma
        term1 = 0.5 * (q_var.log() - p_var.log())
        term2 = 0.5 * ((p_var + (z_mean - g_mean)**2)/q_var - 1.0)
        kl = (term1 + term2).sum(dim=1)
        return kl

    def _compute_local_distribution(self):
        self.model.eval()
        features_dict = {}  # 存储每个类别的特征列表
        with torch.no_grad():
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                # 使用包装模型 forward 得到 z_mean 等
                z_mean, _, _ = self.model(x)
                for i in range(z_mean.size(0)):
                    c = labels[i].item()
                    feat = z_mean[i].cpu()
                    if c not in features_dict:
                        features_dict[c] = []
                    features_dict[c].append(feat)
        local_dist = {}
        for c, feat_list in features_dict.items():
            feats_tensor = torch.stack(feat_list, dim=0)
            mu = feats_tensor.mean(dim=0)
            sigma = feats_tensor.std(dim=0, unbiased=False)
            sigma[sigma < 1e-6] = 1e-6
            local_dist[c] = {"mu": mu, "sigma": sigma}
        self.model.train()
        return local_dist

    def train(self):
        if self.optimizer is None:
            self.init_client()
        self.model.train()

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                z_mean, z_logvar, logits = self.model(x)
                ce_loss = self.criterion(logits, labels).mean()

                kl_loss = 0.0
                if self.global_dist is not None:
                    bs = x.size(0)
                    kl_list = []
                    for i in range(bs):
                        c = labels[i].item()
                        g_mean = self.global_dist[c]["mu"].to(self.device)
                        g_sigma = self.global_dist[c]["sigma"].to(self.device)
                        pm = z_mean[i].unsqueeze(0)
                        pl = z_logvar[i].unsqueeze(0)
                        kl_val = self._kl_gaussian_diag(pm, pl, g_mean, g_sigma)
                        kl_list.append(kl_val)
                    kl_tensor = torch.cat(kl_list, dim=0)
                    kl_loss = kl_tensor.mean()

                loss = ce_loss + self.beta * kl_loss
                loss.backward()
                self.optimizer.step()

        aggregated_state = self.model.base_model.state_dict()
        local_dist = self._compute_local_distribution()

        return {
            "model_state": aggregated_state,
            "local_dist": local_dist,
        }

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                # 因为包装后的模型返回 (z_mean, z_logvar, logits)
                _, _, outputs = self.model(x)
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
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, average="weighted", num_classes=self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            'f1': f1.item()
        }
