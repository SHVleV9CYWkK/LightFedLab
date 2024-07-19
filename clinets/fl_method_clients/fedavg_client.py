from clinets.client import Client


class FedAvgClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)
        self.is_send_gradients = kwargs.get('is_send_gradients', False)

    def train(self):
        self.model.train()

        initial_params = None
        if self.is_send_gradients:
            # 保存初始模型参数的拷贝
            initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        self._local_train()
        if not self.is_send_gradients:
            return self.model.state_dict()

        # 计算从初始到当前的总梯度变化
        total_gradients = {}
        for name, param in self.model.named_parameters():
            if initial_params[name] is not None:
                # 计算总梯度变化
                total_gradient_change = initial_params[name].data - param.data
                total_gradients[name] = total_gradient_change
        return total_gradients
