from servers.server import Server


class FedWCPServer(Server):
    def __init__(self, clients, model, device, args):
        super().__init__(clients, model, device, args)

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)

        sparsity_rate = 0
        for client in self.clients:
            sparsity_rate += client.sparsity_rate

        print('Average sparsity rate:', sparsity_rate / len(self.clients))