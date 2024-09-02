from servers.fl_method_servers.adfedwcp_server import AdFedWCPServer
from servers.fl_method_servers.fedavg_server import FedAvgServer
from servers.fl_method_servers.fedcg_server import FedCGServer
from servers.fl_method_servers.fedmask_server import FedMaskServer
from servers.fl_method_servers.qfedcg_server import QFedCGServer
from servers.fl_method_servers.fedwcp_server import FedWCPServer
from servers.fl_method_servers.pfedgate_server import PFedGateServer
from servers.fl_method_servers.fedem_server import FedEMServer


class ServerFactory:
    def create_server(self, args, clients, model, device):
        param = {
            'server_lr': args.server_lr,
            'n_job': args.n_job,
            'seed': args.seed,
            'client_selection_rate': args.client_selection_rate,
            'optimizer_name': args.optimizer_name,
        }

        fl_type = args.fl_method
        if fl_type == 'fedavg':
            server_prototype = FedAvgServer
        elif fl_type == 'fedcg':
            server_prototype = FedCGServer
        elif fl_type == 'qfedcg':
            server_prototype = QFedCGServer
        elif fl_type == 'fedwcp':
            server_prototype = FedWCPServer
        elif fl_type == 'pfedgate':
            server_prototype = PFedGateServer
        elif fl_type == 'fedmask':
            server_prototype = FedMaskServer
        elif fl_type == 'fedem':
            server_prototype = FedEMServer
        elif fl_type == 'adfedwcp':
            server_prototype = AdFedWCPServer
            param['n_rounds'] = args.n_rounds
            param['k_round'] = args.k_round
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, device, param)