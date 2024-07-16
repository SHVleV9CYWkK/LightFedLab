from servers.server import Server


class FedAvgServer(Server):
    def __init__(self, clients, model, device, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, device, client_selection_rate, server_lr)

    def _average_aggregate(self, weights_list):
        is_send_gradients = self.clients[0].is_send_gradients
        self._gradient_aggregation(weights_list) if is_send_gradients else self._weight_aggregation(weights_list)