import torch
from client import Client


class FedCGClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        self.compression_ratio = kwargs.get('compression_ratio', 1)

    def compress_gradients(self, param_changes):
        # 计算总参数数量和top-k阈值
        numel_list = [p.numel() for p in param_changes.values()]
        total_params = sum(numel_list)
        top_k = int(total_params * self.compression_ratio)

        # 预分配内存来存储所有参数变化量的绝对值及其形状
        all_changes = torch.cat([change.flatten().abs() for change in param_changes.values()])
        _, top_k_indices = torch.topk(all_changes, top_k)

        # 生成压缩后的参数变化量字典
        compressed_param_changes = {}
        idx_offset = 0
        iter_param = iter(param_changes.items())
        for num_elements in numel_list:
            name, change = next(iter_param)
            # 计算当前参数的top-k索引
            current_indices = top_k_indices[(top_k_indices >= idx_offset) & (top_k_indices < idx_offset + num_elements)] - idx_offset
            # 创建一个全零张量
            compressed_change = torch.zeros_like(change).view(-1)
            # 将top-k元素放置回原位置
            compressed_change[current_indices] = change.view(-1)[current_indices]
            compressed_param_changes[name] = compressed_change.view(change.shape)
            idx_offset += num_elements

        return compressed_param_changes

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        param_changes = {name: initial_params[name] - param.data for name, param in self.model.named_parameters()}
        # return self.compress_gradients(param_changes)
        return param_changes