from servers.fedavg_server import FedAvgServer
from servers.fedcg_server import FedCGServer
from servers.qfedcg_server import QFedCGServer
from servers.fedwcp_server import FedWCPServer
from servers.pfedgate_server import PFedGateServer


class ServerFactory:
    def create_server(self, args, clients, model, device):
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
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, device, args.client_selection_rate, args.server_lr)