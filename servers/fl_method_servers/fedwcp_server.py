from tqdm import tqdm

from servers.server import Server


class FedWCPServer(Server):
    def __init__(self, clients, model, device, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, device, client_selection_rate, server_lr)
        print("Initializing client")
        pbar = tqdm(total=len(clients))
        for client in clients:
            client.init_local_model(model)
            pbar.update(1)
        pbar.clear()
        pbar.close()

    def _average_aggregate(self, weights_list):
        self._weight_aggregation(weights_list)