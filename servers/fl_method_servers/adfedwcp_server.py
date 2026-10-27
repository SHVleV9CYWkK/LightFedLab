import math
import random
import sys
import time
import numpy as np
import torch.nn as nn
from servers.fl_method_servers.fedwcp_server import FedWCPServer


class AdFedWCPServer(FedWCPServer):
    # 失真模型 D(k) ≈ c σ² k^-α 的默认参数：高斯分布的 Panter–Dite 常数 √3π/2 与 α = 2，见设计文档式 (7)(8)
    C_F = math.sqrt(3) * math.pi / 2
    ALPHA = 2.0
    # 每个质心 / 每个偏置的传输比特数
    C_W = C_B = 32
    # 对偶变量 μ 二分的相对精度与最大迭代次数
    BISECT_TOL = 1e-6
    BISECT_ITERS = 100

    def __init__(self, clients, model, device, args):
        self.n_rounds = args['n_rounds']
        self.current_rounds = 0
        self.k_ref = args['k_ref']
        self.gamma_a = args['gamma_a']
        self.tau = args['tau']
        self.xi = args['beta']
        self.zeta = args['zeta']
        self.k_min = args['k_min']
        self.k_max = args['k_max']
        self.t_max = args['t_max']
        if not self.k_min <= self.k_ref <= self.k_max:
            raise ValueError(f"k_ref = {self.k_ref} must lie in [k_min, k_max] = [{self.k_min}, {self.k_max}]")
        self.k_candidates = np.arange(self.k_min, self.k_max + 1, dtype=np.float64)

        # 所有 Conv2d / Linear 层；客户端的 imprinting 跳过 downsample 层，其余层按顺序对应 client.layer_importance
        self.layer_keys, self.layer_biases, importance_positions = [], [], []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layer_keys.append(f'{name}.weight')
                self.layer_biases.append(module.bias.numel() if module.bias is not None else 0)
                importance_positions.append(None if 'downsample' in name else
                                            sum(pos is not None for pos in importance_positions))
        # 只有被聚类的层参与质心分配
        state_dict = model.state_dict()
        self.clustered_layers = [j for j, key in enumerate(self.layer_keys)
                                 if clients[0]._should_compress(key, state_dict[key])]
        self.importance_positions = [importance_positions[j] for j in self.clustered_layers]

        # 聚合权重 p_i = D_i / Σ D_l
        total_len = sum(client.train_dataset_len for client in clients)
        self.aggregation_weights = {client.id: client.train_dataset_len / total_len for client in clients}

        # 模拟的上行带宽（Mbps）
        bandwidth_min, bandwidth_max = 5, 100
        mean_bandwidth = (bandwidth_max + bandwidth_min) / 2
        std_bandwidth = (bandwidth_max - bandwidth_min) / 6
        self.bandwidths = {client.id: max(bandwidth_min, min(bandwidth_max,
                                                             round(random.normalvariate(mean_bandwidth, std_bandwidth))))
                           for client in clients}
        print(f"The bandwidth of the clients is: {list(self.bandwidths.values())}")

        self.last_acc, self.current_acc = {}, {}
        super().__init__(clients, model, device, args)
        self.lambda0 = self._calibrate_price()

    def _importance_weights(self, client):
        # ω_{i,j} = L · softmax(I_j / τ)，在有 imprinting 重要性的聚类层上归一化，使其均值为 1，见式 (4)；
        # 没有重要性的 downsample 层取平均敏感度 ω = 1
        omega = np.ones(len(self.clustered_layers))
        ranked = [n for n, pos in enumerate(self.importance_positions) if pos is not None]
        if ranked:
            positions = [self.importance_positions[n] for n in ranked]
            importance = np.asarray(client.layer_importance)[positions] / self.tau
            exp_values = np.exp(importance - importance.max())
            omega[ranked] = len(ranked) * exp_values / exp_values.sum()
        return omega

    def _client_terms(self, client):
        """返回 (聚类层的 key, 惩罚系数 p² ω W c σ², 保留权重数 W, 偏置数 B, 带宽 b（bit/s）)。"""
        keys = [self.layer_keys[j] for j in self.clustered_layers]
        num_biases = np.array([self.layer_biases[j] for j in self.clustered_layers], dtype=np.float64)
        variances = np.array([client.layer_statistics[key][0] for key in keys], dtype=np.float64)
        num_weights = np.array([client.layer_statistics[key][1] for key in keys], dtype=np.float64)
        p = self.aggregation_weights[client.id]
        penalty_scale = p ** 2 * self._importance_weights(client) * num_weights * self.C_F * variances
        return keys, penalty_scale, num_weights, num_biases, self.bandwidths[client.id] * 1e6

    def _calibrate_price(self):
        # λ0 = γ_a · median_{i,j} Λ_{i,j}(k_ref)，Λ 为使最优预算恰为 k_ref 的价格，见式 (17)(18)；只在初始化时计算一次
        prices = []
        for client in self.clients:
            _, penalty_scale, num_weights, _, bandwidth = self._client_terms(client)
            prices.append(self.ALPHA * bandwidth * penalty_scale /
                          (self.C_W * self.k_ref ** (self.ALPHA + 1) + num_weights / math.log(2) * self.k_ref ** self.ALPHA))
        lambda0 = self.gamma_a * float(np.median(np.concatenate(prices)))
        print(f"Calibrated lambda0 = {lambda0:.4g} (k_ref = {self.k_ref}, gamma_a = {self.gamma_a})")
        return lambda0

    def _accuracy_feedback(self, client):
        # η_{i,t}，见式 (13)；没有两轮精度记录时取 1
        if client.id not in self.last_acc or client.id not in self.current_acc:
            return 1.0
        delta_accuracy = self.current_acc[client.id] - self.last_acc[client.id]
        if delta_accuracy > 0:
            return 1 - self.xi * abs(delta_accuracy)
        if delta_accuracy < 0:
            return 1 + self.zeta * abs(delta_accuracy)
        return 1.0

    def _allocate(self, client):
        """为一个客户端求解 (P1)，返回 (各聚类层的 k, 上传时间, 是否为掉队者)。"""
        keys, penalty_scale, num_weights, num_biases, bandwidth = self._client_terms(client)

        # 价格 λ_{i,t} = λ0 / (γ_t η_{i,t})，γ_t = 1 + t / E_max，见式 (12)
        gamma = 1 + self.current_rounds / self.n_rounds
        price = self.lambda0 / (gamma * self._accuracy_feedback(client))

        k = self.k_candidates[None, :]
        # 精度惩罚 S_{i,j}(k) = p² ω W c σ² k^-α，见式 (10)
        penalty = penalty_scale[:, None] * k ** -self.ALPHA
        # 传输时间 T_{i,j}(k) = (C_w k + W log2 k + C_b B) / b，见式 (11)
        upload_time = (self.C_W * k + num_weights[:, None] * np.log2(k) + self.C_B * num_biases[:, None]) / bandwidth
        layer_idx = np.arange(len(keys))

        def solve(mu):
            # 每层的一维问题 (14)：k_max 很小，直接枚举得到精确的整数最优解
            best = np.argmin(penalty + (price + mu) * upload_time, axis=1)
            return self.k_candidates[best].astype(int), upload_time[layer_idx, best].sum()

        k_alloc, total_time = solve(0.0)
        if self.t_max is None or total_time <= self.t_max:
            return k_alloc, total_time, False

        min_time = upload_time[:, 0].sum()
        if min_time > self.t_max:
            return np.full(len(keys), self.k_min), min_time, True

        # 由命题 2，上传时间关于 μ 单调不增：先倍增找到可行上界，再二分最小可行的 μ
        lo, hi = 0.0, max(price, 1e-12)
        while solve(hi)[1] > self.t_max:
            lo, hi = hi, hi * 2
        for _ in range(self.BISECT_ITERS):
            if hi - lo <= self.BISECT_TOL * hi:
                break
            mid = (lo + hi) / 2
            if solve(mid)[1] <= self.t_max:
                hi = mid
            else:
                lo = mid
        k_alloc, total_time = solve(hi)
        return k_alloc, total_time, False

    def calculate_k(self):
        print("Calculating k...")
        start_time = time.time()
        k_lists, upload_times, stragglers = [], [], []
        for client in self.clients:
            k_alloc, upload_time, is_straggler = self._allocate(client)
            client.assign_num_centroids({self.layer_keys[j]: int(k)
                                         for j, k in zip(self.clustered_layers, k_alloc)})
            k_lists.append(k_alloc)
            upload_times.append(upload_time)
            if is_straggler:
                stragglers.append(client.id)
        # 完整打印（每个客户端一行）：numpy 默认超过 1000 个元素就用 "..." 省略，TinyViT / ResNet18 的 k 会丢失
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            print(np.array(k_lists))
        print(f"Upload time per client (s): {np.round(upload_times, 3).tolist()}")
        if stragglers:
            print(f"Stragglers (k_min on every layer still exceeds t_max): {stragglers}")
        print(f"Calculate k time: {time.time() - start_time}s")

    def _collect_client_statistics(self):
        for client in self.selected_clients:
            client.compute_layer_weights()
            client.compute_layer_statistics()

    def train(self):
        self.calculate_k()
        super().train()
        self._collect_client_statistics()
        self.current_rounds += 1

    def evaluate(self):
        result, client_results = super().evaluate()
        self.last_acc = self.current_acc
        self.current_acc = {**self.current_acc,
                            **{client_id: res['accuracy'] for client_id, res in client_results.items()}}
        return result, client_results
