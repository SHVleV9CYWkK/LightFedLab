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

    def _update_wrapper_base_model(self, global_model):
        base_sd = global_model.state_dict()
        self.model.base_model.load_state_dict(base_sd)

    def receive_model(self, global_model):
        if self.model is None:
            hooking_layer = global_model.model.avgpool
            embedding_dim = 512
            self.model = GaussianWrapper(
                base_model=global_model,
                hooking_layer=hooking_layer,
                embedding_dim=embedding_dim,
                latent_dim=128,
                num_classes=self.num_classes
            ).to(self.device)
        else:
            self._update_wrapper_base_model(global_model)

    def _kl_gaussian_diag(self, z_mean, z_logvar, g_mean, g_sigma):
        # 对局部方差 p_var 和全局方差 q_var 做下界截断，确保稳定性
        p_var = torch.clamp(z_logvar.exp(), min=1e-3)
        q_var = torch.clamp(g_sigma * g_sigma, min=1e-3)
        term1 = 0.5 * (q_var.log() - p_var.log())
        term2 = 0.5 * ((p_var + (z_mean - g_mean) ** 2) / q_var - 1.0)
        kl = (term1 + term2).sum(dim=1)
        return kl

    def compute_local_distribution(self):
        self.model.eval()
        features_dict = {}  # 存储每个类别的特征列表
        with torch.no_grad():
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                # 使用包装模型 forward 得到 z_mean
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
            # 对标准差做下界截断，避免过小值（例如由1e-6改为1e-3，可根据情况调整）
            sigma = torch.clamp(feats_tensor.std(dim=0, unbiased=False), min=1e-3)
            local_dist[c] = {"mu": mu, "sigma": sigma, "count": len(feat_list)}
        self.model.train()
        return local_dist

    def train(self):
        if self.optimizer is None:
            self.init_client()
        self.model.train()

        for epoch in range(self.epochs):
            avg_loss = avg_ce_loss = avg_kl_loss = 0

            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                z_mean, z_logvar, logits = self.model(x)
                ce_loss = self.criterion(logits, labels).mean()
                avg_ce_loss += ce_loss.item()

                kl_loss = 0.0
                if self.global_dist is not None:
                    bs = x.size(0)
                    kl_list = []
                    for i in range(bs):
                        c = labels[i].item()
                        # 如果全局分布中没有该类别，则跳过或者可以使用平滑策略
                        if c not in self.global_dist:
                            continue
                        g_mean = self.global_dist[c]["mu"].to(self.device)
                        g_sigma = self.global_dist[c]["sigma"].to(self.device)
                        pm = z_mean[i].unsqueeze(0)
                        pl = z_logvar[i].unsqueeze(0)
                        kl_val = self._kl_gaussian_diag(pm, pl, g_mean, g_sigma)
                        kl_list.append(kl_val)
                    if len(kl_list) > 0:
                        kl_tensor = torch.cat(kl_list, dim=0)
                        kl_loss = kl_tensor.mean()
                    avg_kl_loss += kl_loss

                loss = ce_loss + self.beta * kl_loss
                avg_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Client: {} Epoch: {}, All Loss: {:.4f}, CE loss: {:.4f}, KL Loss: {:.4f}'.format(self.id, epoch, avg_loss, avg_ce_loss, avg_kl_loss))

        aggregated_state = self.model.base_model.state_dict()
        return aggregated_state

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                # 包装模型返回 (z_mean, z_logvar, logits)
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
