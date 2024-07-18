from servers.fl_method_servers.fedavg_server import FedAvgServer
from servers.fl_method_servers.fedcg_server import FedCGServer
from servers.fl_method_servers.qfedcg_server import QFedCGServer
from servers.fl_method_servers.fedwcp_server import FedWCPServer
from servers.fl_method_servers.pfedgate_server import PFedGateServer


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

        optimizer_name = args.optimizer_name if args.is_send_gradients else None
        return server_prototype(clients, model, device, optimizer_name, args.client_selection_rate, args.server_lr)