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
        self.n_samples_mask = hyperparam.get("n_samples_mask", 1)

        self.init_state = None
        self.score_mask_dict = dict()

    def receive_model(self, global_model):
        if self.model is None:
            self.model = deepcopy(global_model).to(self.device)

    def receive_payload(self, payload):
        if self.init_state is None:
            self.init_state = deepcopy(payload["init_state"])
            self.model.load_state_dict(self.init_state)

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
        state = self.model.state_dict()
        for k, w in state.items():
            if k in self.score_mask_dict:
                s = self.score_mask_dict[k]
                mask = sample_mask_with_ste(s)
                new_state[k] = w.to(self.device) * mask
            else:
                new_state[k] = w.to(self.device)

        out = functional_call(self.model, new_state, x)
        return out

    def train(self):
        if self.optimizer is None:
            self.init_client()

        self.model.train()
        for epoch in range(self.epochs):
            avg_loss = 0.
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._forward_with_mask(x)
                loss = self.criterion(outputs, labels).mean()

                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

            avg_loss /= len(self.client_train_loader)
            print(f"[Client {self.id}] Epoch {epoch}, Loss = {avg_loss:.4f}")

        param_dict = self._upload_mask()
        return param_dict

    def _upload_mask(self):
        param_dict = {"mask": {}}

        with torch.no_grad():
            for k, s in self.score_mask_dict.items():
                theta = torch.sigmoid(s.cpu())
                mask_accum = torch.zeros_like(theta)
                for _ in range(self.n_samples_mask):
                    sample = Bernoulli(theta).sample()
                    mask_accum += sample

                param_dict["mask"][k] = mask_accum

        return param_dict

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0.
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self._forward_with_mask(x)
                loss = self.criterion(outputs, labels).mean()
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_labels.append(labels)
                all_preds.append(preds)

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        avg_loss = total_loss / len(self.client_val_loader)
        acc = metrics.multiclass_accuracy(all_preds, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_preds, all_labels, average="weighted",
                                         num_classes=self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': acc.item(),
            'f1': f1.item()
        }
