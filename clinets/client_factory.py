from clinets.fedavg_client import FedAvgClient
from clinets.fedcg_client import FedCGClient
from clinets.qfedcg_client import QFedCGClient
from clinets.fedwcp_client import FedWCPClient


class ClientFactory:
    def create_client(self, num_client, args, dataset_index, full_dataset, criterion, device):
        fl_type = args.fl_method
        if fl_type == 'fedavg':
            client_prototype = FedAvgClient
        elif fl_type == 'fedcg':
            client_prototype = FedCGClient
        elif fl_type == 'qfedcg':
            client_prototype = QFedCGClient
        elif fl_type == 'fedwcp':
            client_prototype = FedWCPClient
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')
        clients = []
        for idx in range(num_client):
            clients.append(client_prototype(idx,
                                            dataset_index[idx],
                                            full_dataset,
                                            args.batch_size,
                                            args.lr,
                                            args.local_epochs,
                                            criterion,
                                            device,
                                            is_send_gradients=args.is_send_gradients,
                                            compression_ratio=args.compression_ratio,
                                            quantization_levels=args.quantization_levels,
                                            reg_lambda=args.reg_lambda,
                                            sparse_compute=args.sparse_compute))

        return clients
