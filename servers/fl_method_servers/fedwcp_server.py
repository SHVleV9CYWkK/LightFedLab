from tqdm import tqdm

from servers.server import Server


class FedWCPServer(Server):
    def __init__(self, clients, model, device, optimizer_name, client_selection_rate=1, server_lr=0.01, n_job=1):
        super().__init__(clients, model, device, optimizer_name, client_selection_rate, server_lr, n_job)

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)
