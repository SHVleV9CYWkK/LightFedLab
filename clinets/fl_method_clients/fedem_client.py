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
        self.actual_train_dataset_len = len(self.client_train_loader.dataset) // hyperparam['bz'] * hyperparam['bz']
        # 初始化后验概率和组件权重
        self.q_t = torch.zeros((self.num_components, self.actual_train_dataset_len), device=self.device)
        self.pi_tm = torch.full((self.num_components,), 1.0 / self.num_components, device=self.device)
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
        # 本地训练
        self._local_train()
        return {m: self.models[m].state_dict() for m in range(self.num_components)}

    def _e_step(self):
        for m in range(self.num_components):
            self.models[m].eval()
            q_temp = torch.zeros(self.actual_train_dataset_len, device=self.device)
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
            self.q_t[m, :] = q_temp
        self.q_t /= self.q_t.sum(dim=0, keepdim=True)

    def _m_step(self):
        pi_new = self.q_t.mean(dim=1)  # 按组件计算平均后验概率
        self.pi_tm = pi_new / pi_new.sum()  # 确保权重和为1

    def _local_train(self):
        self._e_step()
        self._m_step()
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
        # Ensure all learners are in evaluation mode
        for model in self.models:
            model.eval()

        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            # Iterate over each batch from the validation loader
            for i, (x, labels) in enumerate(self.client_val_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                first_output = True

                for m, model in enumerate(self.models):
                    output = model(x)
                    output = F.softmax(output, dim=1)

                    if first_output:
                        outputs = self.pi_tm[m] * output
                        first_output = False
                    else:
                        outputs += self.pi_tm[m] * output

                predicted = outputs.max(1)[1]
                loss = self.criterion(torch.log(outputs), labels)
                loss_meta = loss.mean()
                total_loss += loss_meta.item()

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
