import torch
from servers.server import Server


class FedCRServer(Server):
    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device, args)
        self.global_dist = None
        # 构建客户端ID到其训练数据长度的映射，确保聚合时权重对应正确
        self.client_len = {client.id: client.train_dataset_len for client in self.clients}

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)

    def _merge_distributions(self, local_dists):
        merged_dist = {}
        class_set = set()
        # 1) 收集所有客户端出现过的类别
        for cid, dist_per_class in local_dists.items():
            class_set.update(dist_per_class.keys())

        for c in class_set:
            total_count = 0
            weighted_sum_mu = None

            # ============ 第一轮：先计算全局均值 ============
            for cid, dist_per_class in local_dists.items():
                if c not in dist_per_class:
                    continue
                c_count = dist_per_class[c]['count']
                c_mu = dist_per_class[c]['mu'].to(self.device)

                if weighted_sum_mu is None:
                    weighted_sum_mu = c_mu * c_count
                else:
                    weighted_sum_mu += c_mu * c_count

                total_count += c_count

            if total_count > 0:
                # 全局均值
                global_mu = weighted_sum_mu / total_count

                # ============ 第二轮：计算全局方差 (pooled variance) ============
                pooled_variance = 0.0
                for cid, dist_per_class in local_dists.items():
                    if c not in dist_per_class:
                        continue
                    c_count = dist_per_class[c]['count']
                    c_mu = dist_per_class[c]['mu'].to(self.device)
                    c_sigma = dist_per_class[c]['sigma'].to(self.device)

                    c_var = c_sigma * c_sigma
                    # 加上 (mu_i - global_mu)^2 用来反映各客户端均值与全局均值的差异
                    pooled_variance += c_count * (c_var + (c_mu - global_mu) ** 2)

                pooled_variance = pooled_variance / total_count
                # 对方差做下界，避免出现极小值导致后续KL爆炸
                pooled_variance = torch.clamp(pooled_variance, min=1e-3)
                global_sigma = torch.sqrt(pooled_variance)

                merged_dist[c] = {
                    'mu': global_mu,
                    'sigma': global_sigma
                }
            else:
                # 若本轮没客户端报告该类别，保留之前的全局分布（若存在）
                if self.global_dist is not None and c in self.global_dist:
                    merged_dist[c] = self.global_dist[c]

        return merged_dist

    def _compute_global_dist(self):
        locals_distributions = dict()
        for client in self.selected_clients:
            locals_distributions[client.id] = client.compute_local_distribution()
        self.global_dist = self._merge_distributions(locals_distributions)

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
        self._compute_global_dist()
        self._distribute_model()
        self._distribute_global_dist()
