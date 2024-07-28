from copy import deepcopy
from clinets.client import Client
import torch
import torcheval.metrics.functional as metrics
from utils.utils import get_optimizer, get_lr_scheduler
import torch.nn.functional as F


class FedEMClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.num_components = hyperparam['num_components']
        # 初始化后验概率和组件权重
        self.q_t = [[torch.zeros(self.train_dataset_len, device=self.device)] for _ in range(self.num_components)]
        self.pi_tm = [1.0 / self.num_components for _ in range(self.num_components)]
        self.optimizers = self.lr_schedulers = None
        self.models = [None] * self.num_components

    def receive_model(self, global_models):
        for m in range(self.num_components):
            # 如果是第一次接收模型，则深度复制
            if self.models[m] is None:
                self.models[m] = (deepcopy(global_models[m]).to(device=self.device))
            else:
                self.models[m].load_state_dict(global_models[m].state_dict())

    def init_client(self):
        self.optimizers = []
        for model in self.models:
            optimizer = get_optimizer(self.optimizer_name, model.parameters(), self.lr)
            self.optimizers.append(optimizer)
        # 假设所有模型组件使用相同的学习率调度器设置
        self.lr_schedulers = [get_lr_scheduler(optimizer, self.scheduler_name, self.n_rounds) for optimizer in
                              self.optimizers]

    def update_lr(self, global_metric):
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step(global_metric)

    def train(self):
        # E-step和M-step的实现
        self.e_step()
        self.m_step()

        # 本地训练
        self._local_train()
        return {m: self.models[m].state_dict() for m in range(self.num_components)}

    def e_step(self):
        for m in range(self.num_components):
            self.models[m].eval()
            q_temp = torch.zeros(self.train_dataset_len, device=self.device)
            batch_start = 0
            with torch.no_grad():
                for x, y in self.client_train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.models[m](x)
                    loss = self.criterion(output, y)  # 计算批次的损失
                    # 计算并累积后验概率
                    batch_end = batch_start + x.size(0)
                    q_temp[batch_start:batch_end] = torch.exp(-loss) * self.pi_tm[m]
                    batch_start = batch_end
                self.q_t[m] = q_temp

    def m_step(self):
        total_q = torch.sum(torch.stack([self.q_t[m_prime] for m_prime in range(self.num_components)]))

        # 更新每个组件的权重
        for m in range(self.num_components):
            q_sum = torch.sum(self.q_t[m])
            self.pi_tm[m] = q_sum / total_q

    def _local_train(self):
        for m in range(self.num_components):
            self.models[m].train()
            for epoch in range(self.epochs):
                for i, (x, labels) in enumerate(self.client_train_loader):
                    x, labels = x.to(self.device), labels.to(self.device)
                    self.optimizers[m].zero_grad()  # 清除旧的梯度

                    start_index = i * self.client_train_loader.batch_size
                    end_index = start_index + x.size(0)

                    # 获取当前批次对应的后验概率
                    q_t_batch = self.q_t[m][start_index:end_index].detach()

                    outputs = self.models[m](x)
                    raw_loss = self.criterion(outputs, labels)

                    # 适当处理后验概率
                    q_loss = q_t_batch * raw_loss
                    mean_loss = q_loss.mean()

                    mean_loss.backward()  # 反向传播计算梯度
                    self.optimizers[m].step()  # 更新模型权重

    def evaluate_model(self):
        # 初始化评估指标
        total_loss = 0
        all_labels = []
        all_predictions = []

        # 设置模型为评估模式
        for model in self.models:
            model.eval()

        with torch.no_grad():
            for i, (x, labels) in enumerate(self.client_val_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                start_index = i * self.client_val_loader.batch_size
                end_index = start_index + x.size(0)

                y_pred = torch.zeros((x.size(0), self.num_classes), device=self.device)  # 初始化预测输出
                cumulative_loss = torch.tensor(0., device=self.device)

                for m in range(self.num_components):
                    outputs = self.models[m](x)
                    loss = self.criterion(outputs, labels)
                    q_t_batch = self.q_t[m][start_index:end_index].to(self.device)

                    # 将 loss 扩展到与 q_t_batch 相同的形状
                    adjusted_loss = loss.expand_as(q_t_batch)  # 使用 expand_as 来匹配 q_t_batch 的形状

                    cumulative_loss += q_t_batch * adjusted_loss  # 正确的乘法操作
                    y_pred += q_t_batch.unsqueeze(1) * outputs  # 确保后验概率维度正确并乘以输出

                _, predicted = torch.max(y_pred, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)
                total_loss += cumulative_loss.sum()  # 累加损失

        # 计算总的平均损失和其他统计信息
        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / len(self.client_val_loader)
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, average="weighted", num_classes=self.num_classes)

        return {
            'loss': avg_loss.item(),
            'accuracy': accuracy.item(),
            'f1': f1.item()
        }
