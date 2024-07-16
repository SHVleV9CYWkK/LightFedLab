from servers.server import Server


class FedWCPServer(Server):
    def __init__(self, clients, model, device, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, device, client_selection_rate, server_lr)
        for client in clients:
            client.init_local_model(model)

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)