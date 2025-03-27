import torch
from servers.server import Server


class FedCRServer(Server):
    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device, args)
        self.global_dist = None

    def _average_aggregate(self, weights_list):
        model_states = {}
        for cid, result in weights_list.items():
            model_states[cid] = result["model_state"]

        total_len = sum(self.datasets_len)
        aggregated_state = {}
        # 假设所有客户端返回的 state_dict 格式一致，取任一客户端的 key
        any_cid = next(iter(model_states))
        for key in model_states[any_cid].keys():
            weighted_sum = None
            for i, cid in enumerate(model_states.keys()):
                param = model_states[cid][key].to(self.device)
                weight = self.datasets_len[i]
                if weighted_sum is None:
                    weighted_sum = param * weight
                else:
                    weighted_sum += param * weight
            aggregated_state[key] = weighted_sum / total_len

        self.model.load_state_dict(aggregated_state)

        self.global_dist = self._merge_distributions(weights_list)

    def _merge_distributions(self, weights_list):
        merged_dist = {}
        class_set = set()
        for cid, result in weights_list.items():
            local_dist = result.get("local_dist", {})
            class_set.update(local_dist.keys())

        for c in class_set:
            sum_inv_var = None
            sum_mu_div_var = None
            count = 0
            for cid, result in weights_list.items():
                local_dist = result.get("local_dist", {})
                if c not in local_dist:
                    continue
                mu = local_dist[c]["mu"].to(self.device)
                sigma = local_dist[c]["sigma"].to(self.device)
                var = sigma * sigma
                inv_var = 1.0 / var
                if sum_inv_var is None:
                    sum_inv_var = inv_var
                    sum_mu_div_var = mu * inv_var
                else:
                    sum_inv_var += inv_var
                    sum_mu_div_var += mu * inv_var
                count += 1
            if count > 0:
                global_sigma = torch.sqrt(1.0 / sum_inv_var)
                global_mu = sum_mu_div_var / sum_inv_var
                merged_dist[c] = {"mu": global_mu, "sigma": global_sigma}
        return merged_dist

    def _distribute_global_dist(self):
        if self.global_dist is None:
            return
        for client in self.clients:
            if hasattr(client, "set_global_dist"):
                client.set_global_dist(self.global_dist)

    def train(self):
        print("Training models...")
        clients_weights = self._clients_train()
        print("Aggregating models and distributions...")
        self._average_aggregate(clients_weights)
        self._distribute_model()
        self._distribute_global_dist()
