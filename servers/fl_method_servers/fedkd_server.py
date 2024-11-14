import torch

from servers.server import Server


class FedKDServer(Server):
    def __init__(self, clients, model, device, args):
        """
        FedKD Server初始化
        """
        super().__init__(clients, model, device, args)

    def _average_aggregate(self, weights_list):
        """
        聚合小模型（mentee模型）梯度
        """

        def restore_gradient(svd_gradient):
            """
            从SVD压缩状态恢复梯度，并还原低维和高维张量。

            Args:
                svd_gradient (dict): 压缩状态的SVD梯度。

            Returns:
                dict: 恢复后的梯度。
            """
            restored_gradient = {}
            for key, (u, s, v, original_shape) in svd_gradient.items():
                # 确保奇异值是 1D
                s = s.squeeze()
                if s.ndim != 1:
                    raise ValueError(f"Singular values for {key} are not 1D: {s.shape}")

                # 恢复张量
                restored_tensor = torch.mm(u, torch.mm(torch.diag(s), v.t()))

                # 如果有原始形状，还原为原始形状
                if original_shape is not None:
                    restored_tensor = restored_tensor.view(original_shape)

                restored_gradient[key] = restored_tensor

            return restored_gradient

        # 恢复客户端的梯度
        restored_weights_list = {
            client_id: restore_gradient(weights)
            for client_id, weights in weights_list.items()
        }

        # 聚合梯度并更新模型
        self._gradient_aggregation(restored_weights_list)

    def train(self):
        """
        FedKD Server训练过程
        """
        print("Training models using FedKD...")
        clients_weights = self._clients_train()
        print("Aggregating mentee models...")
        self._average_aggregate(clients_weights)
        self._distribute_model()
