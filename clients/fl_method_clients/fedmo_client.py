from copy import deepcopy

from clients.client import Client


class FedMoClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.global_model = None
        self.base_decay_rate = hyperparam['base_decay_rate']

    def init_client(self):
        self.model = deepcopy(self.global_model).to(device=self.device)
        super().init_client()

    def _compute_global_local_model_difference(self):
        global_dict = self.global_model.state_dict()
        local_dict = self.model.state_dict()
        difference_dict = {}
        for key in global_dict:
            difference_dict[key] = local_dict[key] - global_dict[key]
        return difference_dict

    def train(self):
        ref_momentum = self._compute_global_local_model_difference()

        self.model.train()

        exponential_average_loss = None
        alpha = 0.5  # 损失平衡系数
        for epoch in range(self.epochs):
            for idx, (x, labels) in enumerate(self.client_train_loader):
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

        return self.model.state_dict()
