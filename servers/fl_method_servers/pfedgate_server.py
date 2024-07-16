from models.cnn_model import LeafCNN1
from servers.server import Server


class PFedGateServer(Server):
    def __init__(self, clients, model, device, client_selection_rate=1, server_lr=0.01):
        if not isinstance(model, LeafCNN1):
            raise TypeError("The model must be a LeafCNN1 class or subclass")
        super().__init__(clients, model, device, client_selection_rate, server_lr)
        for client in clients:
            client.init_gating_layer()

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)