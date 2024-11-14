import torch
import torcheval.metrics.functional as metrics
import torch.nn.functional as F
from torch.linalg import svd
from copy import deepcopy
from clients.client import Client

from utils.utils import get_optimizer, get_lr_scheduler


class FedKDClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.mentor_model = None  # 大模型
        self.mentee_model = None  # 小模型
        self.mentee_optimizer = None  # 小模型的优化器
        self.compression_ratio = 1 - kwargs.get('sparse_factor', 0.5)

    def receive_model(self, global_model):
        """
        接收并同步全局Mentee模型。
        """
        if self.mentee_model is None:
            self.mentee_model = deepcopy(global_model).to(device=self.device)
            self.mentor_model = deepcopy(global_model).to(device=self.device)
        else:
            self.mentee_model.load_state_dict(global_model.state_dict())

    def init_client(self):
        """
        初始化客户端，设置优化器和学习率调度器。
        """
        # 初始化Mentor和Mentee的优化器
        self.optimizer = get_optimizer(self.optimizer_name, self.mentor_model.parameters(), self.lr)
        self.mentee_optimizer = get_optimizer(self.optimizer_name, self.mentee_model.parameters(), self.lr)

        # 设置学习率调度器
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    def compress_gradient(self, gradient):
        """
        对梯度进行SVD压缩，并处理低维和高维张量。

        Args:
            gradient (dict): 字典格式的梯度，每个键对应一个张量。

        Returns:
            dict: 压缩后的梯度，每个键仍对应一个张量。
        """
        compressed_gradient = {}
        for name, tensor in gradient.items():
            if tensor is None:  # 跳过无效参数
                continue

            # 确保张量至少是二维
            original_shape = None
            if tensor.ndim > 2:  # 对高维张量展平
                original_shape = tensor.shape
                tensor = tensor.view(tensor.shape[0], -1)  # 展平为二维矩阵 [C_out, C_in * k_h * k_w]
            elif tensor.ndim == 1:  # 对一维张量扩展为二维
                original_shape = tensor.shape
                tensor = tensor.unsqueeze(1)
            elif tensor.ndim == 0:  # 对标量扩展为二维
                original_shape = tensor.shape
                tensor = tensor.unsqueeze(0).unsqueeze(1)

            # 对张量执行 SVD
            u, s, v = torch.linalg.svd(tensor)

            # 确保奇异值是 1D
            if s.ndim != 1:
                raise ValueError(f"Singular values for {name} are not 1D: {s.shape}")

            # 保留前 k 个奇异值
            k = int(len(s) * self.compression_ratio)
            s_compressed = s[:k]
            u_compressed = u[:, :k]
            v_compressed = v[:, :k]

            # 存储压缩结果
            compressed_gradient[name] = (u_compressed, s_compressed, v_compressed, original_shape)

        return compressed_gradient


    def train(self):
        """
        训练过程：大模型和小模型的联合训练，进行相互知识蒸馏。
        """
        self.mentor_model.train()
        self.mentee_model.train()

        initial_params = {name: param.clone() for name, param in self.mentee_model.named_parameters()}

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)

                # Mentor模型的前向传播
                mentor_outputs = self.mentor_model(x)
                mentor_loss = self.criterion(mentor_outputs, labels).mean()

                # Mentee模型的前向传播
                mentee_outputs = self.mentee_model(x)
                mentee_loss = self.criterion(mentee_outputs, labels).mean()

                # 知识蒸馏：Mentor指导Mentee
                kd_loss_mentee = F.kl_div(
                    F.log_softmax(mentee_outputs / 2.0, dim=1),
                    F.softmax(mentor_outputs / 2.0, dim=1),
                    reduction="batchmean",
                )

                # 知识蒸馏：Mentee指导Mentor
                kd_loss_mentor = F.kl_div(
                    F.log_softmax(mentor_outputs / 2.0, dim=1),
                    F.softmax(mentee_outputs / 2.0, dim=1),
                    reduction="batchmean",
                )

                # 总损失
                total_loss = mentor_loss + kd_loss_mentor + mentee_loss + kd_loss_mentee

                # 更新模型
                self.optimizer.zero_grad()
                self.mentee_optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.mentee_optimizer.step()

        # 返回Mentee模型梯度
        param_changes = {name: initial_params[name] - param.data for name, param in self.mentee_model.named_parameters()}
        return self.compress_gradient(param_changes)

    def evaluate_model(self):
        """
        由于个性化评估大模型（Mentor模型）的性能。
        """
        self.mentor_model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.mentor_model(x).to(self.device)
                loss = self.criterion(outputs, labels)
                total_loss += loss.mean()
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
