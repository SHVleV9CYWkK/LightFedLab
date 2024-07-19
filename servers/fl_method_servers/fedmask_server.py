import torch
from tqdm import tqdm
import time

from servers.server import Server


class FedMaskServer(Server):
    def __init__(self, clients, model, device, optimizer_name, client_selection_rate=1, server_lr=0.01):
        super().__init__(clients, model, device, optimizer_name, client_selection_rate, server_lr)
        self.global_masks = None

    def _init_clients(self):
        print("Initializing clients...")
        local_masks = []
        for client in self.clients:
            mask = client.init_client()
            local_masks.append(mask)
            self.global_masks[client.id] = {key: torch.ones_like(value, device=self.device) for key, value in mask}

    def _aggregate_masks(self, client_masks):
        """Aggregate masks from all clients and update each client's personalized mask."""
        for client_id, masks in client_masks.items():
            # Get the current global mask for this client
            global_mask = self.global_masks[client_id]

            # Iterate over each parameter in the mask
            for param_name, client_mask in masks.items():
                # Create a mask that will hold the aggregation result
                aggregation_mask = torch.zeros_like(client_mask)

                # Count how many clients agree on each bit
                agreement_count = torch.zeros_like(client_mask)

                # Check this mask against all other clients' masks for the same parameter
                for other_client_id, other_masks in client_masks.items():
                    if other_client_id != client_id:
                        agreement_count += (client_mask == other_masks[param_name]).int()

                # Determine consensus (more than half of the clients agree on a bit)
                consensus_threshold = len(client_masks) // 2
                consensus_mask = agreement_count > consensus_threshold

                # Apply consensus mask to update global mask for this client
                global_mask[param_name] = consensus_mask.float()

            # Update the global mask for this client
            self.global_masks[client_id] = global_mask

    def _average_aggregate(self, weights_list):
        self._aggregate_masks(weights_list)

    def _distribute_mask(self):
        for client in self.clients:
            client.receive_mask(self.global_masks[client.id])

    def train(self):
        clients_weights = self._clients_train()
        print("Aggregating mask...")
        self._average_aggregate(clients_weights)
        self._distribute_mask()
