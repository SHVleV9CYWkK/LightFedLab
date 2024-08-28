import torch
from clinets.client import Client


class FedCGClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.compression_ratio = 1 - kwargs.get('sparse_factor', 0.5)
        self.gradient_residuals = None

    def compress_gradients(self, param_changes):
        # 计算权重参数的总数量和top-k阈值
        numel_list = [p.numel() for name, p in param_changes.items() if 'bias' not in name]
        total_params = sum(numel_list)
        global_top_k = int(total_params * self.compression_ratio)

        # 生成压缩后的参数变化量字典
        compressed_param_changes = {}
        for name, change in param_changes.items():
            if 'bias' in name:
                compressed_param_changes[name] = change
            else:
                num_elements = change.numel()
                layer_top_k = max(int(num_elements / total_params * global_top_k), 1)
                abs_grad = change.abs().flatten()
                top_values, top_indices = torch.topk(abs_grad, layer_top_k, largest=True, sorted=False)

                mask = torch.zeros_like(change.flatten(), dtype=torch.bool)
                mask[top_indices] = True
                mask = mask.reshape(change.shape)
                self.gradient_residuals[name] = change * (~mask)
                compressed_change = torch.zeros_like(change).view(-1)
                compressed_change[top_indices] = change.view(-1)[top_indices]
                compressed_param_changes[name] = compressed_change.view(change.shape)

        return compressed_param_changes

    def train(self):
        self.model.train()
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        if self.gradient_residuals is None:
            self.gradient_residuals = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels).mean()
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad += self.gradient_residuals[name]
                self.optimizer.step()

        param_changes = {name: initial_params[name] - param.data for name, param in self.model.named_parameters()}
        return self.compress_gradients(param_changes)
