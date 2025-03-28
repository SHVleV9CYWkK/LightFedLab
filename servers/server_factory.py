from servers.fl_method_servers.adfedwcp_server import AdFedWCPServer
from servers.fl_method_servers.fedavg_server import FedAvgServer
from servers.fl_method_servers.fedcg_server import FedCGServer
from servers.fl_method_servers.fedcr_server import FedCRServer
from servers.fl_method_servers.fedkd_server import FedKDServer
from servers.fl_method_servers.fedmask_server import FedMaskServer
from servers.fl_method_servers.fedmo_server import FedMoServer
from servers.fl_method_servers.fedpm_server import FedPMServer
from servers.fl_method_servers.qfedcg_server import QFedCGServer
from servers.fl_method_servers.fedwcp_server import FedWCPServer
from servers.fl_method_servers.pfedgate_server import PFedGateServer
from servers.fl_method_servers.fedem_server import FedEMServer
from servers.fl_method_servers.fedpac_server import FedPACServer


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
            param['beta'] = args.beta
            param['zeta'] = args.zeta
        elif fl_type == 'fedmo':
            server_prototype = FedMoServer
        elif fl_type == 'fedkd':
            server_prototype = FedKDServer
        elif fl_type == 'fedpac':
            server_prototype = FedPACServer
        elif fl_type == 'fedcr':
            server_prototype = FedCRServer
        elif fl_type == 'fedpm':
            server_prototype = FedPMServer
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, device, param)