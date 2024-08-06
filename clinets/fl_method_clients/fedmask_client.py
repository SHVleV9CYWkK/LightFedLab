import torch
from clinets.client import Client
from models.fedmask.mask_model import MaskedModel


class FedMaskClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        self.pruning_rate = kwargs.get('sparse_factor', 0.2)

    def receive_mask(self, masks):
        device = next(self.model.parameters()).device
        masks = {name: mask.to(device) for name, mask in masks.items()}

        updated_masks = {}
        for name, param in self.model.state_dict().items():
            if name not in masks:
                updated_masks[name] = param
            else:
                updated_masks[name] = masks[name] * param.to(device)

        self.model.load_state_dict(updated_masks)


    def train(self):
        self._local_train()
        return self._binarize_mask()

    def _client_pruning(self):
        self._local_train()
        self._prune_mask()
        return self._binarize_mask()

    def _prune_mask(self):
        mask_parameters = {name: param for name, param in self.model.named_parameters() if "mask_" in name}
        for name, mask in mask_parameters.items():
            k = int(self.pruning_rate * mask.numel())
            threshold = torch.topk(mask.view(-1), k, largest=True).values.min()
            mask.data[mask.data < threshold] = 0

    def _binarize_mask(self):
        # Binary mask using sigmoid and threshold of 0.5
        binary_mask = {}
        for name, param in self.model.named_parameters():
            if "mask_" in name:
                binary_mask[name] = (torch.sigmoid(param) > 0.5).float()
        return binary_mask

    def init_client(self):
        self.model = MaskedModel(self.model)
        super().init_client()
        return self._client_pruning()
