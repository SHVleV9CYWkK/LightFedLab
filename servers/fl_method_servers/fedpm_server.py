import torch
from copy import deepcopy
from servers.server import Server


class FedPMServer(Server):
    def __init__(self, clients, model, device, args):
        # 0) 先初始化自己需要的属性
        self.init_state = deepcopy(model.state_dict())

        self.global_score_state = {}
        for k, v in self.init_state.items():
            if "conv" in k or "fc" in k or "weight" in k and "bn" not in k:
                self.global_score_state[k] = torch.zeros_like(v)

        self.lambda_init = args.get("lambda_init", 1.0)
        self.prior_alpha = {}
        self.prior_beta = {}
        for k, w_init in self.init_state.items():
            if "conv" in k or "fc" in k or "weight" in k and "bn" not in k:
                shape = w_init.shape
                self.prior_alpha[k] = torch.ones(shape) * self.lambda_init
                self.prior_beta[k] = torch.ones(shape) * self.lambda_init

        # 1) 再调用父类初始化
        super().__init__(clients, model, device, args)

        # 2) 其它配置
        self.reset_freq = args.get("reset_freq", 5)
        self.round_count = 0

    def _distribute_model(self):
        payload = {
            "init_state": self.init_state,
            "score_state": self.global_score_state
        }
        for client in self.clients:
            client.receive_model(self.model)
            client.receive_payload(payload)

    def _average_aggregate(self, weights_list):
        # 定期重置
        if (self.round_count > 0) and (self.round_count % self.reset_freq == 0):
            for k in self.prior_alpha:
                if "conv" in k or "fc" in k or "weight" in k and "bn" not in k:
                    self.prior_alpha[k].fill_(self.lambda_init)
                    self.prior_beta[k].fill_(self.lambda_init)

        # 累加
        for k in self.init_state.keys():
            if "conv" in k or "fc" in k or "weight" in k and "bn" not in k:
                alpha_ = self.prior_alpha[k].to(self.device)
                beta_ = self.prior_beta[k].to(self.device)
                for cid, param_dict in weights_list.items():
                    if "mask" not in param_dict:
                        continue
                    if k not in param_dict["mask"]:
                        continue
                    mask_sum = param_dict["mask"][k].to(self.device)
                    alpha_ += mask_sum
                    beta_ += (1.0 - mask_sum)
                self.prior_alpha[k] = alpha_.cpu()
                self.prior_beta[k] = beta_.cpu()

        # 更新global_score
        new_score_state = {}
        for k in self.init_state.keys():
            if "conv" in k or "fc" in k or "weight" in k and "bn" not in k:
                alpha_ = self.prior_alpha[k]
                beta_ = self.prior_beta[k]
                numerator = alpha_ - 1.0
                denominator = (alpha_ + beta_ - 2.0).clamp_min(1e-8)
                tmp = (numerator / denominator).clamp(1e-5, 1 - 1e-5)
                score_ = torch.log(tmp / (1.0 - tmp))
                new_score_state[k] = score_.detach().cpu()
        self.global_score_state = new_score_state

    def train(self):
        self._sample_clients()
        clients_weights = self._clients_train()
        self._average_aggregate(clients_weights)
        self.round_count += 1
        self._distribute_model()
