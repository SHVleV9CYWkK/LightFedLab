from clinets.fedavg_client import FedAvgClient
from clinets.fedcg_client import FedCGClient
from clinets.qfedcg_client import QFedCGClient
from clinets.fedcc_client import FedCCClient


class ClientFactory:
    def create_client(self, num_client, args, dataset_index, full_dataset, criterion, device):
        fl_type = args.fl_method
        is_gradients = args.is_send_gradients
        fedcg_cr = args.compression_ratio
        qfedcg_ql = args.quantization_levels
        fedcc_lambda = args.reg_lambda
        if fl_type == 'fedavg':
            client_prototype = FedAvgClient
        elif fl_type == 'fedcg':
            client_prototype = FedCGClient
        elif fl_type == 'qfedcg':
            client_prototype = QFedCGClient
        elif fl_type == 'fedcc':
            client_prototype = FedCCClient
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
                                            is_send_gradients=is_gradients,
                                            compression_ratio=fedcg_cr,
                                            quantization_levels=qfedcg_ql,
                                            reg_lambda=fedcc_lambda))

        return clients
