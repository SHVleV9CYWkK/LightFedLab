from clinets.client import Client


class FedAvgClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device)

    def train(self):
        self._local_train()
        return self.model.state_dict()
