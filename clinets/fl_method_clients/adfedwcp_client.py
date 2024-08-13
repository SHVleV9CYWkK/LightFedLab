import torch
from clinets.fl_method_clients.fedwcp_client import FedWCPClient
from utils.kmeans import TorchKMeans


class AdFedWCPClient(FedWCPClient):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, **kwargs)
        self.num_centroids = {}
        self.layer_importance = {}
        enable_interlayers = kwargs.get('enable_interlayers', False)
        self.assign_num_centroids = self.calculate_num_centroids_interlayers \
            if enable_interlayers else self.assign_num_centroids_to_interlayers
        if enable_interlayers:
            self.compute_layer_importance()

    def compute_layer_importance(self):
        pass

    def calculate_num_centroids_interlayers(self, k):
        pass

    def assign_num_centroids_to_interlayers(self, k):
        for key, weight in self.model.state_dict().items():
            if 'weight' in key:
                self.num_centroids[key] = k
                self.layer_importance[key] = 1

    def set_num_centroids(self, k):
        self.assign_num_centroids(k)

    def _cluster_and_prune_model_weights(self):
        clustered_state_dict = {}
        mask_dict = {}
        for key, weight in self.model.state_dict().items():
            if 'weight' in key:
                original_shape = weight.shape
                kmeans = TorchKMeans(n_clusters=self.num_centroids[key], is_sparse=True)
                flattened_weights = weight.detach().view(-1, 1)
                kmeans.fit(flattened_weights)

                new_weights = kmeans.centroids[kmeans.labels_].view(original_shape)
                is_zero_centroid = (kmeans.centroids == 0).view(-1)
                mask = is_zero_centroid[kmeans.labels_].view(original_shape) == 0
                mask_dict[key] = mask.bool()
                clustered_state_dict[key] = new_weights
            else:
                clustered_state_dict[key] = weight
                mask_dict[key] = torch.ones_like(weight, dtype=torch.bool)
        return clustered_state_dict, mask_dict
