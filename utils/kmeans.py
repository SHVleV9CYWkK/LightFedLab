import torch


class TorchKMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=300, tol=5e-3,
                 batch_size=4096, is_sparse=False,
                 enforce_2of4=False):

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.centroids = None
        self.labels_ = None
        self.is_sparse = is_sparse
        # 新增一个标志，用于决定是否执行2:4稀疏化后处理
        self.enforce_2of4 = enforce_2of4

    def fit(self, X):
        """
        X 通常是 [n_samples, n_features] 的Tensor；
        这里的 n_samples 表示“被聚类的实体数量”，
        n_features 表示每个实体的特征维度。
        在模型权重压缩场景下，往往把每个权重当做一个sample（或小片段）。
        具体需你自行 reshape/flatten 后再传进来。
        """
        # 先对输入做个中心化
        X -= X.mean(axis=0)
        best_loss = float('inf')
        n_samples = X.size(0)

        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X)
            prev_loss = float('inf')
            loss = None

            for _ in range(self.max_iter):
                x = X
                if self.batch_size is not None:
                    # 随机选一个batch做迭代
                    minibatch_indices = torch.randperm(n_samples, device=X.device)[:self.batch_size]
                    x = X[minibatch_indices]

                # 计算与质心的距离并分配标签
                distances = torch.cdist(x, centroids)
                labels = torch.argmin(distances, dim=1)

                # one-hot
                labels_one_hot = torch.nn.functional.one_hot(labels, self.n_clusters).type_as(x)

                # 如果 is_sparse = True，则假设第0个centroid是0向量，其余是非零质心
                if self.is_sparse:
                    # 跳过第0列，因为它代表 0 质心（不会更新）
                    labels_one_hot_for_centroids = labels_one_hot[:, 1:]
                    new_centroids = labels_one_hot_for_centroids.t().matmul(x) / (
                            labels_one_hot_for_centroids.sum(0)[:, None] + 1e-10
                    )
                    # 更新除第0个以外的质心
                    centroids[1:] = new_centroids
                else:
                    # 普通情况，所有质心都要更新
                    centroids = labels_one_hot.t().matmul(x) / (
                            labels_one_hot.sum(0)[:, None] + 1e-10
                    )

                min_distances = distances[torch.arange(distances.size(0)), labels]
                loss = (min_distances ** 2).sum()
                r = torch.abs((prev_loss - loss) / (prev_loss + 1e-9))
                if r < self.tol:
                    break
                prev_loss = loss

            # 记录最优质心
            if loss < best_loss:
                best_loss = loss
                self.centroids = centroids

        # 在整体数据上计算最终labels
        all_distances = torch.cdist(X, self.centroids)
        self.labels_ = torch.argmin(all_distances, dim=1)

        # -- NEW CODE for 2:4 enforce (可选) --
        if self.is_sparse and self.enforce_2of4:
            # 对每4个元素强制2个归0质心
            self._enforce_2of4_sparsity_vectorized(X)

        return self

    def _initialize_centroids(self, X):
        """
        根据 is_sparse 决定：若 is_sparse=True 则第一个质心设为0向量，剩下的用kmeans++方式。
        """
        if len(X) > 4194304:
            n_samples = 4194304
            indices = torch.randperm(len(X), device=X.device)[:n_samples]
            X_subsample = X[indices]
        else:
            X_subsample = X

        centroids = []
        if self.is_sparse:
            zero_centroid = torch.zeros_like(X_subsample[0])
            centroids.append(zero_centroid)
        else:
            # 普通情况，第一个质心随机初始化
            first_index = torch.randint(0, len(X_subsample), (1,))
            centroids.append(X_subsample[first_index].squeeze(0))

        # 其余质心使用 kmeans++ 方式初始化
        for _ in range(1, self.n_clusters):
            centroids_tensor = torch.stack(centroids)
            distances = torch.cdist(X_subsample, centroids_tensor).min(dim=1)[0] + 1e-10
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_index = torch.multinomial(probabilities, 1)
            centroids.append(X_subsample[next_index].squeeze(0))

        return torch.stack(centroids).to(X.device)

    def _enforce_2of4_sparsity(self, X):
        """
        针对 self.labels_ 强制：每连续4个元素中，恰好有2个为0质心（label=0），2个非0质心。
        注意：这里假定 is_sparse=True 时，第0个centroid就是 0 向量。

        X: shape [n_samples, n_features]
        self.labels_: shape [n_samples]

        说明：
          - 在很多真实场景下，你可能对 Flatten 后的一维权重进行分块。
          - 这里为了演示，假设 X 的每一行只有 1 维特征（n_features=1），
            或者你先把模型权重 flatten 成 [n_total_params, 1].
          - 若你想对 [n_samples, n_features>1] 进行2:4 enforce，需要在行或列维度上再做reshape/遍历。
        """
        # 先检查X形状
        n_samples, n_features = X.shape
        if n_features != 1:
            print("[警告] 目前的示例仅在 n_features=1 时方便演示 2:4 block，请根据实际需求修改。")

        # 将 X 和 labels 都视作一维展开方便处理
        data_1d = X.view(-1)
        labels_1d = self.labels_.view(-1)  # shape相同, 都是 n_samples

        total_len = data_1d.shape[0]
        zero_label = 0  # 我们假设第0个质心就是 0 向量

        for i in range(0, total_len, 4):
            end_idx = min(i + 4, total_len)

            # 如果剩余不足4个元素, 直接跳过
            if end_idx > total_len:
                # 也可以写成 if (end_idx - i) < 4: break
                # 这里使用 break，说明后面的都不处理
                break

            block_indices = range(i, end_idx)

            # 统计本block中有多少元素被分配到 0质心
            block_labels = labels_1d[block_indices]
            zero_mask = (block_labels == zero_label)
            num_zero = zero_mask.sum().item()

            # 如果已经是2个0，2个非0，则无须处理
            if num_zero == 2:
                continue

            # 如果 zero>2，则需要减少一些0
            elif num_zero > 2:
                diff = num_zero - 2
                # 选出 block 中那diff个“最不该为0”的元素，强制改为离它最近的非零质心
                # 这里举例：找出非零质心距离最小/或数据值绝对值较大的优先改回非零
                # 简单做法：按照 data_1d 的绝对值从大到小排序
                zero_indices = [j for j, is0 in zip(block_indices, zero_mask) if is0]
                zero_vals = torch.abs(data_1d[zero_indices])
                # 找到最Top diff个
                # 这里简化处理: 直接选绝对值最大的diff个去改为非零
                _, sorted_idx = torch.sort(zero_vals, descending=True)
                to_flip = sorted_idx[:diff]
                flip_global_indices = [zero_indices[idx.item()] for idx in to_flip]

                # 给这些 flip_global_indices 的 label 改成“最近的非零质心”
                self._assign_nearest_nonzero_centroid(data_1d, labels_1d, flip_global_indices)

            # 如果 zero<2，则需要增加一些0
            elif num_zero < 2:
                diff = 2 - num_zero
                # 选出 block 中“最应该为0”的 diff 个元素，改为0 label
                # 可以基于与0质心的距离(其实就是数值本身绝对值越小，越应该是0)
                # 简化写法：我们取绝对值最小的diff个 => 强制标签=0
                nonzero_indices = [j for j, is0 in zip(block_indices, zero_mask) if not is0]
                nonzero_vals = torch.abs(data_1d[nonzero_indices])
                _, sorted_idx = torch.sort(nonzero_vals, descending=True)  # 或者ascending=True也可
                # 如果想把最小值变0，应该 ascending=True 取前diff个
                # 这里演示一下descending=False
                _, sorted_idx2 = torch.sort(nonzero_vals, descending=False)
                to_flip = sorted_idx2[:diff]
                flip_global_indices = [nonzero_indices[idx.item()] for idx in to_flip]

                # 直接把这些位置的label改为0
                for gi in flip_global_indices:
                    labels_1d[gi] = zero_label

        # 最后将修改后的labels再赋值回 self.labels_
        self.labels_ = labels_1d.view_as(self.labels_)

    def _assign_nearest_nonzero_centroid(self, data_1d, labels_1d, flip_indices):
        """
        对一批本来是0 label的元素，重新分配到非零质心中。
        """
        # 先计算这些位置与各质心的距离，找最近的非零质心
        # 注意：self.centroids.shape = [n_clusters, n_features]
        # 此时 data_1d 是一维，但若 n_features>1，需要按行对齐。
        # 简单起见，假设 n_features=1
        if self.centroids is None:
            return
        zero_label = 0
        nonzero_labels = range(1, self.n_clusters)  # 1~n_clusters-1
        sub_data = data_1d[flip_indices].unsqueeze(1)  # shape [X, 1]
        cdist = torch.cdist(sub_data, self.centroids)  # shape [X, n_clusters]

        # 找到最小距离对应的centroid
        nearest_centroids = torch.argmin(cdist, dim=1)
        # 如果那个centroid == 0，则要再找第二小？ 简化处理：直接在 nonzero_labels 中找最小
        # 下面示例做法：屏蔽第0列距离，再找最小
        cdist[:, zero_label] = 1e10  # 把0质心的距离置很大
        nearest_nonzeros = torch.argmin(cdist, dim=1)

        for i, gi in enumerate(flip_indices):
            labels_1d[gi] = nearest_nonzeros[i]

    def _enforce_2of4_sparsity_vectorized(self, X):
        device = X.device
        n_samples, n_features = X.shape
        if n_features != 1:
            raise ValueError("当前示例仅支持 n_features=1 的情况")
        if n_samples % 4 != 0:
            raise ValueError("数据长度不是4的倍数")

        labels_2d = self.labels_.view(-1, 4)    # [n_block, 4]
        zero_mask_2d = (labels_2d == 0)         # [n_block, 4]
        zero_count = zero_mask_2d.sum(dim=1)    # [n_block]

        needs_more_zeros = zero_count < 2       # [n_block]
        needs_less_zeros = zero_count > 2       # [n_block]

        data_2d = X.view(-1, 4)                # [n_block, 4]
        abs_data_2d = data_2d.abs()            # [n_block, 4]

        # 处理需要增加零的块 (非0 -> 0)
        if needs_more_zeros.any():
            blocks_to_add = torch.where(needs_more_zeros)[0]
            zeros_needed = 2 - zero_count[needs_more_zeros]  # [M]

            # 对非零位置的值进行排序
            nonzero_mask = ~zero_mask_2d[blocks_to_add]  # [M, 4]
            values = abs_data_2d[blocks_to_add]          # [M, 4]
            values[~nonzero_mask] = float('inf')         # 屏蔽已经是零的位置

            # 每块选择最小的k个非零值的位置
            _, indices = torch.topk(values, k=4, dim=1, largest=False)  # [M, 4]

            # 创建正确维度的flip mask
            num_blocks = len(blocks_to_add)
            # 扩展 zeros_needed 到 [M, 4] 维度
            zeros_needed_expanded = zeros_needed.unsqueeze(1).expand(-1, 4)  # [M, 4]
            # 创建范围张量 [M, 4], 每行是 [0,1,2,3]
            pos_range = torch.arange(4, device=device).expand(num_blocks, -1)  # [M, 4]

            # 创建正确维度的掩码 [M, 4]
            flip_mask = pos_range < zeros_needed_expanded

            # 获取需要翻转的位置
            flip_positions = indices[flip_mask]  # [K], K是总共需要翻转的位置数

            # 计算全局索引
            # 重复blocks_to_add以匹配flip_positions的大小
            block_offsets = torch.repeat_interleave(blocks_to_add,
                                                    zeros_needed.to(torch.long)) * 4
            global_indices = block_offsets + flip_positions

            # 设置为零
            self.labels_.view(-1)[global_indices] = 0

        # 处理需要减少零的块 (0 -> 非0)
        if needs_less_zeros.any() and self.centroids is not None:
            blocks_to_remove = torch.where(needs_less_zeros)[0]
            zeros_to_remove = zero_count[blocks_to_remove] - 2  # [N]

            # 对零位置的值进行排序
            zero_mask = zero_mask_2d[blocks_to_remove]  # [N, 4]
            values = abs_data_2d[blocks_to_remove]      # [N, 4]
            values[~zero_mask] = float('-inf')          # 屏蔽非零位置

            # 选择最大的k个零值的位置
            _, indices = torch.topk(values, k=4, dim=1, largest=True)  # [N, 4]

            # 创建正确维度的flip mask，类似上面的处理
            num_blocks = len(blocks_to_remove)
            zeros_to_remove_expanded = zeros_to_remove.unsqueeze(1).expand(-1, 4)  # [N, 4]
            pos_range = torch.arange(4, device=device).expand(num_blocks, -1)      # [N, 4]
            flip_mask = pos_range < zeros_to_remove_expanded                       # [N, 4]

            # 获取需要翻转的位置
            flip_positions = indices[flip_mask]  # [L], L是总共需要翻转的位置数

            # 计算全局索引
            block_offsets = torch.repeat_interleave(blocks_to_remove,
                                                    zeros_to_remove.to(torch.long)) * 4
            global_indices = block_offsets + flip_positions

            # 为这些位置找到最近的非零质心
            if len(global_indices) > 0:
                flip_data = X.view(-1)[global_indices].unsqueeze(1)  # [L, 1]
                nonzero_centroids = self.centroids[1:]              # [C-1, 1]

                # 计算距离
                a_norm = (flip_data ** 2).sum(dim=1, keepdim=True)    # [L, 1]
                b_norm = (nonzero_centroids ** 2).sum(dim=1)          # [C-1]
                dist = a_norm + b_norm - 2 * torch.matmul(flip_data, nonzero_centroids.t())

                nearest_idx = torch.argmin(dist, dim=1)  # [L]
                self.labels_.view(-1)[global_indices] = nearest_idx + 1

        return self.labels_