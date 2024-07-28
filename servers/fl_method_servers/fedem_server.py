import copy

from servers.server import Server


class FedEMServer(Server):
    def __init__(self, clients, model, device, optimizer_name, seed, client_selection_rate=1, server_lr=0.01, n_job=1):
        self.num_components = clients[0].num_components
        self.models = [copy.deepcopy(model) for _ in range(self.num_components)]
        super().__init__(clients, model, device, optimizer_name, seed, client_selection_rate, server_lr, n_job)

    def _distribute_model(self):
        for client in self.clients:
            client.receive_model(self.models)

    def _average_aggregate(self, weights_list):
        datasets_len = self.datasets_len if self.is_all_clients else [client.dataset_len for client in
                                                                      self.selected_clients]
        total_len = sum(datasets_len)

        for m in range(self.num_components):
            average_weights = {}
            for key in weights_list[0][m].keys():
                weighted_sum = sum(weights_list[client_id][m][key].to(self.device) * datasets_len[client_id]
                                   for client_id in range(len(weights_list)))
                average_weights[key] = weighted_sum / total_len
            self.models[m].load_state_dict(average_weights)
