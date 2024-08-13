import torch

from servers.server import Server


class FedMaskServer(Server):
    def __init__(self, clients, model, device, args):
        self.global_masks = dict()
        super().__init__(clients, model, device, args)

    def _init_clients(self):
        print("Initializing clients...")
        local_masks = []
        for client in self.clients:
            mask = client.init_client()
            local_masks.append(mask)
            self.global_masks[client.id] = {key: torch.ones_like(value, device=self.device)
                                            for key, value in mask.items()}

    def _aggregate_masks(self):
        # Aggregate only elements that appear in two or more masks
        aggregated_masks = {}

        # First, count how many times each element in each parameter is set to 1 across all clients
        mask_count = {}
        for client_masks in self.global_masks.values():
            for param_name, mask in client_masks.items():
                if param_name not in mask_count:
                    mask_count[param_name] = torch.zeros_like(mask)
                mask_count[param_name] += mask

        # Determine which elements appear in more than one mask
        for param_name, count in mask_count.items():
            # Create a mask where elements that appear in at least two masks are 1, others are 0
            # This boolean mask will determine which elements to aggregate
            aggregation_criteria = count >= 2
            if param_name not in aggregated_masks:
                aggregated_masks[param_name] = torch.zeros_like(count)

            # Aggregate only the elements that meet the criteria
            for client_id, client_masks in self.global_masks.items():
                aggregated_masks[param_name] += client_masks[param_name] * aggregation_criteria

            # Average the values that were aggregated
            # Avoid division by zero by using maximum with ones tensor
            element_count = torch.max(torch.tensor(1.0), aggregation_criteria.float() * count)
            aggregated_masks[param_name] /= element_count

        # Update the global masks based on aggregation result
        for client_id in self.global_masks:
            for param_name in self.global_masks[client_id]:
                # Only update the elements that were actually aggregated
                self.global_masks[client_id][param_name] = aggregated_masks[param_name]

    def _average_aggregate(self, weights_list):
        for idx, weights in weights_list.items():
            self.global_masks[idx] = weights
        self._aggregate_masks()

    def _distribute_mask(self):
        for client in self.clients:
            client.receive_mask(self.global_masks[client.id])

    def train(self):
        clients_weights = self._clients_train()
        print("Aggregating mask...")
        self._average_aggregate(clients_weights)
        self._distribute_mask()
