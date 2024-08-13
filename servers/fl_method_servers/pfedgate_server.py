from models.cnn_model import LeafCNN1
from servers.server import Server


class PFedGateServer(Server):
    def __init__(self, clients, model, device, args):
        if not isinstance(model, LeafCNN1):
            raise TypeError("The model must be a LeafCNN1 class or subclass")
        super().__init__(clients, model, device, args)
        for client in clients:
            client.init_gating_layer()

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)

    def evaluate(self):
        results = super().evaluate()
        self.clients[0].global_metric = results["accuracy"]
        return results
