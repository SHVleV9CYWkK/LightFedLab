import torch
import torch.nn as nn
from copy import deepcopy
from torch.distributions import Bernoulli
from torch.func import functional_call

import torcheval.metrics.functional as metrics
from clients.client import Client

from utils.utils import get_optimizer, get_lr_scheduler


def sample_mask_with_ste(score):
    theta = torch.sigmoid(score)
    rand_tensor = torch.rand_like(theta)
    binary_mask = (theta > rand_tensor).float()
    mask = binary_mask.detach() + theta - theta.detach()
    return mask


class FedPMClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, dl_n_job=0, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, dl_n_job)
        self.lr = hyperparam.get("lr", 0.01)
        self.epochs = hyperparam.get("local_epochs", 1)
        self.n_samples_mask = hyperparam.get("n_samples_mask", 1)
        self.eps = hyperparam.get("eps", 1e-2)

        self.resnet_init_state = None
        self.score_mask_dict = dict()

    def receive_model(self, global_model):
        if self.model is None:
            self.model = deepcopy(global_model).to(self.device)

    def receive_payload(self, payload):
        if self.resnet_init_state is None:
            self.resnet_init_state = deepcopy(payload["init_state"])
            self.model.load_state_dict(self.resnet_init_state)

        # 同步 score
        if "score_state" in payload:
            for k, v in payload["score_state"].items():
                if k not in self.score_mask_dict:
                    param = nn.Parameter(v.clone().float().to(self.device), requires_grad=True)
                    self.score_mask_dict[k] = param
                else:
                    self.score_mask_dict[k].data = v.clone().float().to(self.device)

    def init_client(self):
        param_list = []
        for k, v in self.score_mask_dict.items():
            v.requires_grad = True
            param_list.append(v)
        self.optimizer = get_optimizer(self.optimizer_name, param_list, self.lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    def _forward_with_mask(self, x):
        new_state = {}
        for k, w_init in self.resnet_init_state.items():
            # 若是“conv/linear”的weight, 并且在 score_mask_dict 中
            if k in self.score_mask_dict and ("conv" in k or "fc" in k or ("weight" in k and "bn" not in k)):
                s = self.score_mask_dict[k]
                mask = sample_mask_with_ste(s)
                new_state[k] = w_init.to(self.device) * mask
            else:
                # 其余如 bias, BN.running_xxx, BN.weight/bias 直接用 w_init
                new_state[k] = w_init.to(self.device)
        # 利用 new_state 做一次函数式 forward, 不会修改 model 里原数据
        out = functional_call(self.model, new_state, x)
        return out

    def train(self):
        if self.optimizer is None:
            self.init_client()

        self.model.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._forward_with_mask(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                self.optimizer.step()

        param_dict, local_freq, num_bits = self._upload_mask()
        return param_dict

    def _upload_mask(self):
        param_dict = {"mask": dict()}
        num_params_total = 0
        sum_ones = 0

        with torch.no_grad():
            for k, s in self.score_mask_dict.items():
                if "conv" in k or "fc" in k or ("weight" in k and "bn" not in k):
                    s_cpu = s.cpu()
                    theta = torch.sigmoid(s_cpu)
                    mask_accum = torch.zeros_like(theta)
                    for i in range(self.n_samples_mask):
                        # Bernoulli 随机采样
                        bern = Bernoulli(theta)
                        sample = bern.sample()
                        # eps 处理
                        sample = torch.where(sample < 0.5,
                                             torch.tensor(self.eps),
                                             torch.tensor(1.0 - self.eps))
                        mask_accum += sample
                        sum_ones += (sample > 0.5).sum().item()
                        num_params_total += sample.numel()

                    param_dict["mask"][k] = mask_accum  # n_samples_mask 次之和

        freq_ones = sum_ones / (num_params_total)
        return param_dict, freq_ones, num_params_total

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self._forward_with_mask(x)
                loss = self.criterion(outputs, labels).mean()
                total_loss += loss

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
