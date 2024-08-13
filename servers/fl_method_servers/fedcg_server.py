from servers.server import Server


class FedCGServer(Server):
    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device,args)

    def _average_aggregate(self, weights_list):
        self._gradient_aggregation(weights_list)
