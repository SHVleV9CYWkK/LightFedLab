from servers.server import Server


class pFedMeServer(Server):
    """pFedMe 服务器：w^{t+1} = (1 - β) w^t + β · avg(w_i)，β = 1 时即 FedAvg 的模型平均，见 Algorithm 1 第 10 行。"""

    def __init__(self, clients, model, device, args):
        self.beta = args['pfedme_beta']
        super().__init__(clients, model, device, args)

    def _average_aggregate(self, weights_list):
        previous_state = {key: value.clone() for key, value in self.model.state_dict().items()}
        self._weight_aggregation(weights_list)
        if self.beta == 1:
            return
        mixed_state = {}
        for key, averaged in self.model.state_dict().items():
            if averaged.is_floating_point():
                mixed_state[key] = (1 - self.beta) * previous_state[key] + self.beta * averaged
            else:
                mixed_state[key] = averaged
        self.model.load_state_dict(mixed_state)
