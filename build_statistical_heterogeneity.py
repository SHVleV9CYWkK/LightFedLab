import os
import numpy as np
import torch
from itertools import cycle
from utils.args import *
from sklearn.model_selection import train_test_split
from utils.utils import *


def split_data_with_dirichlet(num_clients, a, dataset, test_size, seed):
    # 分配数据到客户端
    data_indices = [np.array([]) for _ in range(num_clients)]
    min_size = 0
    while min_size < 10:  # 确保每个客户端至少有10个样本
        # 为每个类别生成Dirichlet分布
        for label in range(len(dataset.classes)):
            label_indices = np.where(np.array(dataset.targets) == label)[0]
            # 使用Dirichlet分布分配这个类别的索引
            distributed_indices = np.random.dirichlet(np.repeat(a, num_clients)) * len(label_indices)
            distributed_indices = np.cumsum(distributed_indices).astype(int)
            data_indices = [np.concatenate((data_indices[i], label_indices[
                                                             distributed_indices[i - 1]:distributed_indices[
                                                                 i]])) if i != 0 else np.concatenate(
                (data_indices[i], label_indices[:distributed_indices[i]])) for i in range(num_clients)]
        min_size = min([len(i) for i in data_indices])
        data_indices = [i.astype(int) for i in data_indices]

        # 为每个客户端划分验证集
    train_val_split = {}
    for i in range(num_clients):
        train_idx, val_idx = train_test_split(data_indices[i], test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_idx, 'val': val_idx}
    return train_val_split


def split_data_with_label(number_clients, number_client_label, dataset, test_size, seed):
    number_labels = len(torch.unique(dataset.targets))

    if number_client_label > number_labels:
        raise ValueError("number_client_label cannot be greater than the total number of labels.")

    label_pool = np.tile(np.arange(number_labels), (number_client_label, 1)).T.flatten()
    np.random.shuffle(label_pool)

    label_to_clients = {i: [] for i in range(number_clients)}
    label_pool_cycle = cycle(label_pool)
    for i in range(number_clients):
        while len(label_to_clients[i]) < number_client_label:
            next_label = next(label_pool_cycle)
            if next_label not in label_to_clients[i]:
                label_to_clients[i].append(next_label)

    # 初始化为列表，每个客户端一个空列表
    clients_data_indices = [[] for _ in range(number_clients)]
    for label, clients in label_to_clients.items():
        label_indices = np.where(np.array(dataset.targets) == label)[0]
        split_indices = np.array_split(label_indices, len(clients))
        for i, client in enumerate(clients):
            clients_data_indices[client].extend(split_indices[i].tolist())

    # 转换每个客户端的列表为ndarray
    train_val_split = {}
    for i in range(len(clients_data_indices)):
        clients_data_indices[i] = np.array(clients_data_indices[i])
        train_idx, val_idx = train_test_split(clients_data_indices[i], test_size=test_size, random_state=seed)
        train_val_split[i] = {'train': train_idx, 'val': val_idx}

    return train_val_split


def save_client_indices(dir, dataset_name, split_method, indexes):
    os.makedirs(dir, exist_ok=True)
    dataset_subdir = os.path.join(dir, dataset_name+"_"+split_method)
    os.makedirs(dataset_subdir, exist_ok=True)

    for client_id, client_data in indexes.items():
        dataset_subdir_client = os.path.join(dataset_subdir, f"client_{client_id}")
        os.makedirs(dataset_subdir_client, exist_ok=True)
        train_file = os.path.join(dataset_subdir_client, "train_indexes.npy")
        val_file = os.path.join(dataset_subdir_client, f"val_indexes.npy")
        np.save(train_file, client_data['train'])
        np.save(val_file, client_data['val'])
        print(f"Saved training indices for client {client_id} at {train_file}")
        print(f"Saved validation indices for client {client_id} at {val_file}")


if __name__ == "__main__":
    args = parse_args_for_dataset()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    full_dataset = load_dataset(args.dataset_name)
    if args.split_method == "dirichlet":
        indices = split_data_with_dirichlet(args.clients_num, args.alpha, full_dataset, args.test_ratio, args.seed)
    elif args.split_method == "label":
        indices = split_data_with_label(args.clients_num, args.number_label, full_dataset, args.test_ratio, args.seed)
    else:
        raise ValueError(f"split_method does not contain {args.split_method}")

    save_client_indices(args.dataset_indexes_dir, args.dataset_name, args.split_method, indices)
    print(f"Saved client indices to {args.dataset_indexes_dir}")