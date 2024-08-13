from clinets.fl_method_clients.fedavg_client import FedAvgClient
from clinets.fl_method_clients.fedcg_client import FedCGClient
from clinets.fl_method_clients.fedmask_client import FedMaskClient
from clinets.fl_method_clients.qfedcg_client import QFedCGClient
from clinets.fl_method_clients.fedwcp_client import FedWCPClient
from clinets.fl_method_clients.adfedwcp_client import AdFedWCPClient
from clinets.fl_method_clients.pfedgate_client import PFedGateClient
from clinets.fl_method_clients.fedem_client import FedEMClient


class ClientFactory:
    def create_client(self, num_client, args, dataset_index, full_dataset, device):
        train_hyperparam = {
            "optimizer_name": args.optimizer_name,
            "lr": args.lr,
            "bz": args.batch_size,
            "local_epochs": args.local_epochs,
            "scheduler_name": args.scheduler_name,
            "n_rounds": args.n_rounds
        }

        fl_type = args.fl_method
        if fl_type == 'fedavg':
            client_prototype = FedAvgClient
        elif fl_type == 'fedcg':
            client_prototype = FedCGClient
        elif fl_type == 'qfedcg':
            client_prototype = QFedCGClient
        elif fl_type == 'fedwcp':
            client_prototype = FedWCPClient
        elif fl_type == 'pfedgate':
            client_prototype = PFedGateClient
            train_hyperparam['gating_lr'] = args.gating_lr
        elif fl_type == 'fedmask':
            client_prototype = FedMaskClient
        elif fl_type == 'fedem':
            client_prototype = FedEMClient
            train_hyperparam['num_components'] = args.num_components
        elif fl_type == 'adfedwcp':
            client_prototype = AdFedWCPClient
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        clients = []
        for idx in range(num_client):
            clients.append(client_prototype(idx,
                                            dataset_index[idx],
                                            full_dataset,
                                            train_hyperparam,
                                            device,
                                            quantization_levels=args.quantization_levels,
                                            sparse_factor=args.sparse_factor,
                                            dl_n_job=args.dl_n_job))

        return clients
