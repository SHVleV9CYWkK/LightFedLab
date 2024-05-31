import random
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

    # def _save_log(self, eval_results):
    #     # 获取今天的日期字符串
    #     today_date = datetime.today().strftime('%Y-%m-%d')
    #
    #     # 创建今天日期的目录
    #     log_dir = os.path.join(self.save_log_dir, today_date)
    #     os.makedirs(log_dir, exist_ok=True)
    #
    #     # 遍历评估结果，保存到相应的文件中
    #     for metric, value in eval_results.items():
    #         file_path = os.path.join(log_dir, f"{metric}.txt")
    #         with open(file_path, 'a') as file:
    #             file.write(f"{value}\n")
    #
    #     print(f"Evaluation results saved to {log_dir}")

    def _sample_clients(self):
        if self.client_selection_rate != 1:
            self.selected_clients = random.sample(self.clients, int(len(self.clients) * self.client_selection_rate))
        else:
            self.selected_clients = self.clients

    def train(self):
        self._sample_clients()
        for client in self.selected_clients:
            client.train()
        self._average_aggregate()
        self._distribute_model()
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