import numpy as np
import torch
import torcheval.metrics.functional as metrics
from clinets.client import Client
from models.pfedgate.gating_layers import GatingLayer
from models.pfedgate.knapsack_solver import KnapsackSolver01
from utils.utils import get_lr_scheduler, get_optimizer


class PFedGateClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, hyperparam, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, hyperparam, device, kwargs.get('dl_n_job', 0))
        data_sample, _ = full_dataset[0]
        self.dataset_name = full_dataset.__class__.__name__.lower()
        self.input_feat_size = data_sample.numel()
        self.num_channels = data_sample.size(0)
        self.gating_layer = None
        self.gating_lr = hyperparam['gating_lr']
        self.min_sparse_factor = self.total_model_size = None
        self.knapsack_solver = None
        self.sparse_factor = kwargs.get('sparse_factor', 0.5)
        self.gated_scores_scale_factor = 10
        self.opt_for_gating = None
        self.lr_scheduler_for_gating = self.lr_scheduler = None

    def init_gating_layer(self):
        self.gating_layer = GatingLayer(self.model, self.device, self.dataset_name, 5, 0)
        self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
        self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, 0.1), self.sparse_factor)
        self.knapsack_solver = KnapsackSolver01(
            value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
            item_num_max=len(self.gating_layer.block_size_lookup_table),
            weight_max=round(self.sparse_factor * self.total_model_size.item())
        )
        self.opt_for_gating = get_optimizer(self.optimizer_name, self.gating_layer.parameters(), lr=self.gating_lr)
        self.lr_scheduler_for_gating = get_lr_scheduler(self.opt_for_gating, self.scheduler_name, self.n_rounds, True)

    def _get_top_gated_scores(self, x):
        """ Get gating weights via the learned gating layer data-dependently """
        # get gating weights data-dependently via gumbel trick
        gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
        # normed importance score
        gated_scores = torch.sigmoid(gating_logits)
        gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

        # separate trans
        if id(gated_scores) != id(trans_weights):
            # bounded model diff
            trans_weights = torch.sigmoid(trans_weights)
            trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

        # avoid cutting info flow (some internal sub-blocks are all zeros)
        gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

        top_trans_weights, sparse_ratio_selected = self._select_top_trans_weights(gated_scores, trans_weights)

        return gated_scores, top_trans_weights, sparse_ratio_selected

    def _adapt_prune_model(self, top_trans_weights):
        """

        """

        if self.gating_layer.fine_grained_block_split == 1:
            for para_idx, para in enumerate(self.model.parameters()):
                mask = torch.ones_like(para, device=self.device).reshape(-1) * top_trans_weights[para_idx]
                para_name = self.gating_layer.block_names[para_idx]
                mask = mask.view(para.shape)
                self.model.set_adapted_para(para_name, mask * para)
        else:
            for para_name, para in self.model.named_parameters():
                mask = torch.ones_like(para, device=self.device).reshape(-1)
                sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                    para_name]
                for i in range(sub_block_begin, sub_block_end):
                    gating_weight_sub_block_i = top_trans_weights[i]
                    block_element_begin = (i - sub_block_begin) * size_each_sub
                    block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                    mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                mask = mask.view(para.shape)
                self.model.set_adapted_para(para_name, mask * para)
        return top_trans_weights.detach()

    def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
        weight_list = self.gating_layer.block_size_lookup_table[block_idx]
        importance_value_list = importance_value_list[block_idx]
        capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
        total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = (
            self.knapsack_solver.found_max_value_greedy(
                weight_list=weight_list.tolist(),
                value_list=importance_value_list,
                capacity=capacity
            ))

        droped_item_idx = np.array(block_idx)[droped_item_idx]
        mask[droped_item_idx] *= 0

        if isinstance(total_weight, torch.Tensor):
            return total_weight.detach()
        else:
            return total_weight

    def _select_top_trans_weights(self, gated_scores, trans_weight):
        """

        """

        if self.sparse_factor == 1:
            return trans_weight, torch.tensor(1.0)

        retained_trans_weights = trans_weight

        # keep top (self.sparse_factor) weights via 0-1 knapsack
        mask = torch.ones_like(gated_scores, device=self.device)
        if id(trans_weight) != id(gated_scores):
            # ST trick
            mask = mask - gated_scores.detach() + gated_scores
        importance_value_list = np.array(gated_scores.tolist())
        importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(int)

        # for linear_layer sub_blocks
        linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
        selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                    mask)

        # for non-linear-layer sub_blocks
        non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
        selected_size += self._select_top_sub_blocks(importance_value_list, non_linear_layer_block_idx_filter_first,
                                                     mask)

        retained_trans_weights *= mask

        return retained_trans_weights, selected_size / self.total_model_size

    def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
        """

        """

        weight_list = self.gating_layer.block_size_lookup_table[block_idx]
        importance_value_list = importance_value_list[block_idx]
        capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
        total_value_of_selected_items, selected_items_weight, selected_items_frac = (
            self.knapsack_solver.found_max_value(
                weight_list=weight_list * (1 - self.min_sparse_factor),
                value_list=importance_value_list,
                capacity=capacity
            ))
        gated_scores_after_select[block_idx] += selected_items_weight / weight_list

        return sum(selected_items_weight).detach()

    def fit_batch(self, x, y):
        """

        """
        self.opt_for_gating.zero_grad()
        self.optimizer.zero_grad()

        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        # get personalized masks via gating layer
        _, top_trans_weights, _ = self._get_top_gated_scores(x)
        # # mask the meta-model according to sparsity preference
        _ = self._adapt_prune_model(top_trans_weights)

        y_pred = self.model.adapted_forward(x)
        loss_vec = self.criterion(y_pred, y)
        loss_meta_model = loss_vec.mean()
        loss_meta_model.backward()

        self.opt_for_gating.step()
        self.optimizer.step()

        self.model.del_adapted_para()

        return loss_meta_model

    def update_lr(self, global_metric):
        self.lr_scheduler.step(global_metric)
        self.lr_scheduler_for_gating.step(global_metric)

    def train(self):
        self.model.train()
        self.gating_layer.train()
        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                _ = self.fit_batch(x, labels)

        return self.model.state_dict()

    def evaluate_model(self):
        self.model.eval()
        self.gating_layer.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for x, labels in self.client_val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                _, top_trans_weights, _ = self._get_top_gated_scores(x)
                _ = self._adapt_prune_model(top_trans_weights)
                outputs = self.model.adapted_forward(x)
                loss_vec = self.criterion(outputs, labels)
                loss_meta_model = loss_vec.mean()
                total_loss += loss_meta_model
                _, predicted = torch.max(outputs.data, 1)
                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        avg_loss = total_loss / len(self.client_val_loader)
        accuracy = metrics.multiclass_accuracy(all_predictions, all_labels, num_classes=self.num_classes)
        # precision = metrics.multiclass_precision(all_predictions, all_labels, num_classes=self.num_classes)
        # recall = metrics.multiclass_recall(all_predictions, all_labels, num_classes=self.num_classes)
        f1 = metrics.multiclass_f1_score(all_predictions, all_labels, average="weighted", num_classes=self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            # 'precision': precision.item(),
            # 'recall': recall.item(),
            'f1': f1.item()
        }
