import time
import random
from tqdm import tqdm
from abc import ABC, abstractmethod

class Server(ABC):
    def __init__(self, clients, model, client_selection_rate=1):
        self.clients = clients
        self.selected_clients = None
        self.client_selection_rate = client_selection_rate
        self.is_all_clients = client_selection_rate == 1
        self.model = model
        self.datasets_len = [client.dataset_len for client in self.clients]
        self._distribute_model()

    @abstractmethod
    def _average_aggregate(self):
        pass

    def _distribute_model(self):
        for client in self.clients if self.is_all_clients else self.selected_clients:
            client.receive_model(self.model)

    def _evaluate_model(self):
        result_list = []
        for client in self.selected_clients:
            result = client.evaluate_local_model()
            result_list.append(result)

        metrics_keys = result_list[0].keys()

        average_results = {key: 0 for key in metrics_keys}

        for result in result_list:
            for key in metrics_keys:
                average_results[key] += result.get(key, 0)

        for key in average_results.keys():
            average_results[key] /= len(result_list)

        return average_results

    def _sample_clients(self):
        if self.client_selection_rate != 1:
            self.selected_clients = random.sample(self.clients, int(len(self.clients) * self.client_selection_rate))
        else:
            self.selected_clients = self.clients

    def train(self):
        self._sample_clients()
        pbar = tqdm(total=len(self.selected_clients))
        for client in self.selected_clients:
            client.train()
            pbar.update(1)
        pbar.clear()
        pbar.close()
        print("Aggregating models")
        start_time = time.time()
        self._average_aggregate()
        end_time = time.time()
        print(f"Aggregation takes {(end_time - start_time):.3f} seconds")
        self._distribute_model()
        print("Evaluating model")
        average_eval_results = self._evaluate_model()
        return average_eval_results


class FedAvgServer(Server):
    def __init__(self, clients, model, client_selection_rate=1):
        super().__init__(clients, model, client_selection_rate)

    def _average_aggregate(self):
        weights_list = [client.model.state_dict()
                        for client in (self.clients if self.is_all_clients else self.selected_clients)]
        datasets_len = self.datasets_len if self.is_all_clients else [client.dataset_len for client in self.selected_clients]
        average_weights = {}
        for key in weights_list[0].keys():
            weighted_sum = sum(weights[key] * len_ for weights, len_ in zip(weights_list, datasets_len))
            total_len = sum(datasets_len)
            average_weights[key] = weighted_sum / total_len

        self.model.load_state_dict(average_weights)