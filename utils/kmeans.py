# 基于Pytorch的K-Means实现类
import torch


class TorchKMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=300, tol=5e-3, batch_size=4096, is_sparse=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.centroids = None
        self.labels_ = None
        self.is_sparse = is_sparse

    def fit(self, X):
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
                    minibatch_indices = torch.randperm(n_samples, device=X.device)[:self.batch_size]
                    x = X[minibatch_indices]

                distances = torch.cdist(x, centroids)
                labels = torch.argmin(distances, dim=1)

                labels_one_hot = torch.nn.functional.one_hot(labels, self.n_clusters).type_as(x)
                if self.is_sparse:
                    labels_one_hot = labels_one_hot[:, 1:]
                    new_centroids = labels_one_hot.t().matmul(x) / (labels_one_hot.sum(0)[:, None] + 1e-10)
                    centroids[1:] = new_centroids
                else:
                    centroids = labels_one_hot.t().matmul(x) / (labels_one_hot.sum(0)[:, None] + 1e-10)
                min_distances = distances[torch.arange(distances.size(0)), labels]
                loss = (min_distances ** 2).sum()
                r = torch.abs((prev_loss - loss) / prev_loss)
                if r < self.tol:
                    break
                prev_loss = loss

            if loss < best_loss:
                best_loss = loss
                self.centroids = centroids

        all_distances = torch.cdist(X, self.centroids)
        self.labels_ = torch.argmin(all_distances, dim=1)

        return self

    def _initialize_centroids(self, X):
        n_samples = 4194204 if len(X) > 4194204 else len(X)
        # 随机选择索引
        indices = torch.randperm(len(X))[:n_samples]

        # 创建子样本
        X_subsample = X[indices]
        centroids = []

        if self.is_sparse:
            zero_centroid = torch.zeros_like(X_subsample[0])
            centroids.append(zero_centroid)
        else:
            first_index = torch.randint(0, len(X_subsample), (1,))
            centroids.append(X_subsample[first_index].squeeze(0))

        for _ in range(1, self.n_clusters):
            centroids_tensor = torch.stack(centroids)
            distances = torch.cdist(X_subsample, centroids_tensor).min(dim=1)[0] + 1e-10
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_index = torch.multinomial(probabilities, 1)
            centroids.append(X_subsample[next_index].squeeze(0))

        return torch.stack(centroids).to(X.device)
