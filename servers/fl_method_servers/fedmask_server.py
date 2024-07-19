import torch
from tqdm import tqdm
import time

from servers.server import Server


class FedMaskServer(Server):
    def __init__(self, clients, model, device, optimizer_name, client_selection_rate=1, server_lr=0.01):
        self.global_masks = dict()
        super().__init__(clients, model, device, optimizer_name, client_selection_rate, server_lr)

    def _init_clients(self):
        print("Initializing clients...")
        local_masks = []
        for client in self.clients:
            mask = client.init_client()
            local_masks.append(mask)
            self.global_masks[client.id] = {key: torch.ones_like(value, device=self.device)
                                            for key, value in mask.items()}

    def _aggregate_masks(self):
        """Aggregate masks from all clients based on consensus."""
        # Initialize the count of agreement for each parameter across all clients
        mask_agreement_count = {}
        consensus_threshold = len(self.global_masks) // 2

        # Initialize the agreement count dictionary
        for client_id, masks in self.global_masks.items():
            for param_name, mask in masks.items():
                if param_name not in mask_agreement_count:
                    mask_agreement_count[param_name] = torch.zeros_like(mask)
                # Sum up the masks from all clients for each parameter
                mask_agreement_count[param_name] += mask

        # Update the global masks based on the consensus count
        for client_id, masks in self.global_masks.items():
            for param_name, global_mask in masks.items():
                # Apply the consensus check to update the global mask
                consensus_mask = mask_agreement_count[param_name] > consensus_threshold
                self.global_masks[client_id][param_name] = consensus_mask.float()

    def _average_aggregate(self, weights_list):
        for id, weights in weights_list.items():
            self.global_masks[id] = weights
        self._aggregate_masks()

    def _distribute_mask(self):
        for client in self.clients:
            client.receive_mask(self.global_masks[client.id])

    def train(self):
        clients_weights = self._clients_train()
        print("Aggregating mask...")
        self._average_aggregate(clients_weights)
        self._distribute_mask()
