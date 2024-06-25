from servers.fedavg_server import FedAvgServer
from servers.fedcg_server import FedCGServer
from servers.qfedcg_server import QFedCGServer
from servers.fedwcp_server import FedWCPServer


class ServerFactory:
    def create_server(self, fl_type, clients, model, device, client_selection_rate=1, server_lr=1e-3):
        if fl_type == 'fedavg':
            server_prototype = FedAvgServer
        elif fl_type == 'fedcg':
            server_prototype = FedCGServer
        elif fl_type == 'qfedcg':
            server_prototype = QFedCGServer
        elif fl_type == 'fedcc':
            server_prototype = FedWCPServer
        else:
            raise NotImplementedError(f'Invalid Federated learning method name: {fl_type}')

        return server_prototype(clients, model, device, client_selection_rate, server_lr)