import torch
from clinets.client import Client
from models.pfedgate.gating_layers import GatingLayer
from models.pfedgate.knapsack_solver import KnapsackSolverFractional
from models.pfedgate.nn_nets import DifferentiableCeilFun, DifferentiableRoundFun


class PFedGateClient(Client):
    def __init__(self, client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device, **kwargs):
        super().__init__(client_id, dataset_index, full_dataset, bz, lr, epochs, criterion, device)
        data_sample, _ = full_dataset[0]
        self.input_feat_size = data_sample.numel()
        self.num_channels = data_sample.size(0)
        self.gating_layer = None
        self.min_sparse_factor = self.total_model_size = None
        self.knapsack_solver = None
        self.sparse_factor = kwargs.get('sparse_factor', 0.5)

    def init_gating_layer(self):
        self.gating_layer = GatingLayer(self.model, self.device, self.input_feat_size, self.num_channels)
        self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
        self.min_sparse_factor = min(max(1 / self.gating_layer.fine_grained_block_split, 0.1), self.sparse_factor)
        self.knapsack_solver = KnapsackSolverFractional(
            item_num_max=len(self.gating_layer.block_size_lookup_table),
            weight_max=round(self.sparse_factor * self.total_model_size.item())
        )

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_gating_layer = torch.optim.Adam(self.gating_layer.parameters(), lr=1e-6)
        initial_model_params = {name: param.clone() for name, param in self.model.named_parameters()}

        for epoch in range(self.epochs):
            for x, labels in self.client_train_loader:
                x, labels = x.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                optimizer_gating_layer.zero_grad()
                gating_weights = self.gating_layer(x)
                self._prune_model_weights(gating_weights)  # 应用稀疏化
                outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer_gating_layer.step()

        total_model_gradients = {}
        for name, param in self.model.named_parameters():
            if initial_model_params[name] is not None:
                # 计算总梯度变化
                total_gradient_change = initial_model_params[name].data - param.data
                total_model_gradients[name] = total_gradient_change

        return total_model_gradients

    def _prune_model_weights(self, gating_weights):
        pass
        # gating_scores, trans_weights = gating_weights  # 解构得到两种权重
        # # 对权重应用 Sigmoid 函数进行规范化，以确保所有得分都在0和1之间
        # gating_scores = torch.sigmoid(gating_scores)
        # trans_weights = torch.sigmoid(trans_weights)
        #
        # # 取均值减少批次维度影响，得到每个块的代表性权重
        # gating_scores = torch.mean(gating_scores, dim=0)  # -> [Num_blocks]
        # trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]
        #
        # idx = 0  # 初始化索引，用于访问与参数块对应的权重
        # for name, param in self.model.named_parameters():
        #     # 获取当前参数对应的权重
        #     current_gating_score = gating_scores[idx]
        #     current_trans_weight = trans_weights[idx]
        #
        #     # 计算当前参数的掩码，这里结合两个得分
        #     combined_score = (current_gating_score + current_trans_weight) / 2
        #
        #     # 扩展掩码以匹配参数形状
        #     mask = combined_score.expand_as(param)
        #
        #     # 应用掩码，根据得分进行稀疏化
        #     # 实际阈值可能需要根据模型和任务需求进行调整
        #     param.data = torch.where(mask > 0.1, param.data, torch.zeros_like(param.data))
        #
        #     idx += 1  # 移动到下一个参数块

    def adapt_prune_model(self, top_trans_weights):
        """

        """

        ceil_fun, round_fun = DifferentiableCeilFun.apply, DifferentiableRoundFun.apply
        if self.gating_layer.fine_grained_block_split == 1:
            for para_idx, para in enumerate(self.model.parameters()):
                ori_size, total_para_num = para.shape, para.numel()
                mask = torch.ones_like(para, device=self.device).reshape(-1) * top_trans_weights[para_idx]
                # mask = ceil_fun(mask)  # ones Tensor while keep the grad
                # select at most the first gated_scores[para_idx] parameters as each dim
                mask[round_fun(total_para_num * top_trans_weights[para_idx]):] = 0
                mask = mask.view(ori_size)
                para_name = self.gating_layer.block_names[para_idx]
                # self.model.adapted_model_para[para_name] = mask * para
                self.model.set_adapted_para(para_name, mask * para)
        else:
            for para_name, para in self.model.named_parameters():
                mask = torch.ones_like(para, device=self.device).reshape(-1)
                sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                    para_name]
                for i in range(sub_block_begin, sub_block_end):
                    block_element_begin = (i - sub_block_begin) * size_each_sub
                    # select the first gating_weight_sub_block_i para
                    block_element_end_selected = round_fun(
                        (i + 1 - sub_block_begin) * size_each_sub * top_trans_weights[i])
                    block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                    mask[block_element_begin:block_element_end_selected] *= top_trans_weights[i]
                    mask[block_element_end_selected:block_element_end] = 0
                mask = mask.view(para.shape)
                # self.model.adapted_model_para[para_name] = mask * para
                self.model.set_adapted_para(para_name, mask * para)
        return top_trans_weights.detach()

    def _select_top_trans_weights(self, gating_weights):
        """

        """

        gated_scores, trans_weight = gating_weights
        if self.sparse_factor == 1:
            return trans_weight, torch.tensor(1.0)

        retained_trans_weights = trans_weight

        trans_weights_after_select = torch.full_like(trans_weight, fill_value=self.min_sparse_factor)

        # for linear_layer sub_blocks
        linear_layer_block_idx = self.gating_layer.linear_layer_block_idx
        selected_size = self._select_top_sub_blocks_frac(gated_scores, linear_layer_block_idx,
                                                         trans_weights_after_select)

        # for non-linear-layer sub_blocks
        non_linear_layer_block_idx = self.gating_layer.non_linear_layer_block_idx
        selected_size += self._select_top_sub_blocks_frac(retained_trans_weights, non_linear_layer_block_idx,
                                                          trans_weights_after_select)

        calibration_quantity = trans_weights_after_select - trans_weight.detach()
        retained_trans_weights += calibration_quantity

        return retained_trans_weights, selected_size / self.total_model_size

    def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
        """

        """

        weight_list = self.gating_layer.block_size_lookup_table[block_idx]
        importance_value_list = importance_value_list[block_idx]
        capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
        total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
            weight_list=weight_list * (1 - self.min_sparse_factor),
            value_list=importance_value_list,
            capacity=capacity
        )
        gated_scores_after_select[block_idx] += selected_items_weight / weight_list

        return sum(selected_items_weight).detach()
