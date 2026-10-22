from copy import deepcopy
import torch
from clients.client import Client
from utils.utils import get_lr_scheduler


class pFedMeClient(Client):
    """
    pFedMe (Dinh et al., NeurIPS 2020) 客户端。

    self.model 是个性化模型 θ_i，self.local_model 是本地模型 w_{i,r}。每个 mini-batch：
      1. 用 K 步梯度下降近似求解 θ_i = argmin f_i(θ; D) + λ/2 ||θ - w||²，见论文式 (7)
      2. w ← w - η λ (w - θ_i)，见 Algorithm 1 第 8 行
    上传 w；评估时同时报告个性化模型 θ_i（主指标）与全局模型 w 的结果。
    """

    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.lam = hyperparam['pfedme_lambda']
        self.inner_steps = hyperparam['pfedme_k']
        self.personal_lr = hyperparam['personal_lr']
        self.local_model = None

    def receive_model(self, global_model):
        # θ_i 只在第一次接收时初始化，之后保留上一轮的个性化结果用于评估
        if self.model is None:
            super().receive_model(global_model)
        if self.local_model is None:
            self.local_model = deepcopy(global_model).to(device=self.device)
        else:
            self.local_model.load_state_dict(global_model.state_dict())

    def init_client(self):
        # 内层问题按论文用梯度下降求解，学习率调度作用于内层步长
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.personal_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.scheduler_name, self.n_rounds)

    def train(self):
        # 每轮的个性化模型从 w^t 出发，并在本轮的各次外层更新间热启动
        self.model.load_state_dict(self.local_model.state_dict())
        self.model.train()
        local_params = list(self.local_model.parameters())

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                for _ in range(self.inner_steps):
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.model(x), labels).mean()
                    proximal = sum(((theta - w.detach()) ** 2).sum()
                                   for theta, w in zip(self.model.parameters(), local_params))
                    (loss + self.lam / 2 * proximal).backward()
                    self.optimizer.step()

                with torch.no_grad():
                    for w, theta in zip(local_params, self.model.parameters()):
                        w.sub_(self.lr * self.lam * (w - theta))

        # BatchNorm 等缓冲区只在个性化模型的前向中更新，上传前同步给本地模型
        with torch.no_grad():
            for w_buffer, theta_buffer in zip(self.local_model.buffers(), self.model.buffers()):
                w_buffer.copy_(theta_buffer)
        return self.local_model.state_dict()

    def evaluate_model(self):
        result = super().evaluate_model()
        personalized_model, self.model = self.model, self.local_model
        try:
            global_result = super().evaluate_model()
        finally:
            self.model = personalized_model
        result.update({f'global_{key}': value for key, value in global_result.items()})
        return result
