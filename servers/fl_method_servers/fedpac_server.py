import torch
from servers.server import Server  # 你提供的基类


class FedPACServer(Server):

    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device, args)

        # 这里可以存放全局特征中心
        # 假设形状: [num_classes, feature_dim]
        self.global_centroids = None

        # 如果需要对服务器端的分类器也做初始化，可以在这儿处理
        # 比如 self.global_classifier = deepcopy(model.classifier.state_dict()) # 视情况而定

    def _average_aggregate(self, weights_list):
        rep_state_dicts = {}
        clf_state_dicts = {}

        for cid, full_sd in weights_list.items():
            local_rep = {}
            local_clf = {}

            # 遍历全部 key
            for k, v in full_sd.items():
                if 'fc' in k:  # 以fc作为分类器区分
                    local_clf[k] = v
                else:
                    local_rep[k] = v

            rep_state_dicts[cid] = local_rep
            clf_state_dicts[cid] = local_clf

        if self.is_all_clients:
            datasets_len = self.datasets_len
        else:
            datasets_len = [client.train_dataset_len for client in self.selected_clients]

        total_len = sum(datasets_len)
        rep_aggregated = {}

        any_cid = next(iter(rep_state_dicts.keys()))
        for key in rep_state_dicts[any_cid].keys():
            # 进行加权
            weighted_sum = None
            for i, cid in enumerate(rep_state_dicts.keys()):
                w = rep_state_dicts[cid][key].to(self.device)
                scale = datasets_len[i]  # client i
                if weighted_sum is None:
                    weighted_sum = w * scale
                else:
                    weighted_sum += w * scale
            rep_aggregated[key] = weighted_sum / total_len

        current_global_sd = self.model.state_dict()
        for k in rep_aggregated:
            if k in current_global_sd:  # 只更新特征提取器部分
                current_global_sd[k] = rep_aggregated[k]

        if hasattr(self.model, 'num_classes'):
            num_classes = self.model.num_classes
        else:
            num_classes = self.clients[0].num_classes

        centroid_sums = torch.zeros((num_classes, rep_aggregated[next(iter(rep_aggregated))].shape[0]),
                                    device=self.device)
        centroid_counts = torch.zeros(num_classes, device=self.device)

        idx = 0
        for cid in weights_list:
            local_c = weights_list[cid].get("local_centroids", None)
            if local_c is None:
                continue
            c_len = datasets_len[idx]  # client idx
            centroid_sums += local_c.to(self.device) * float(c_len)
            centroid_counts += float(c_len)
            idx += 1

        for clz in range(num_classes):
            if centroid_counts[clz] > 0:
                centroid_sums[clz] /= centroid_counts[clz]
        self.global_centroids = centroid_sums

        clf_aggregated = {}
        any_cid2 = next(iter(weights_list.keys()))
        local_clf_sd = clf_state_dicts[any_cid2]  # 先做 clone
        # 跟 rep aggregator 类似
        for key in local_clf_sd.keys():
            weighted_sum = None
            for i, cid in enumerate(weights_list.keys()):
                w = clf_state_dicts[cid][key].to(self.device)
                scale = datasets_len[i]
                if weighted_sum is None:
                    weighted_sum = w * scale
                else:
                    weighted_sum += w * scale
            clf_aggregated[key] = weighted_sum / total_len

        # 直接把聚合后的分类器参数也更新到 self.model
        for k in clf_aggregated:
            if k in current_global_sd:
                current_global_sd[k] = clf_aggregated[k]

        # 把更新后的 state_dict load 回全局模型
        self.model.load_state_dict(current_global_sd)

    def train(self):
        print("Training models...")
        clients_weights = self._clients_train()

        print("Aggregating models...")
        self._average_aggregate(clients_weights)

        self._distribute_model()

        # 还要分发全局特征中心
        self._distribute_global_centroids()

    def _distribute_global_centroids(self):
        if self.global_centroids is None:
            return

        for client in self.clients:
            if hasattr(client, "set_global_centroids"):
                client.set_global_centroids(self.global_centroids)
