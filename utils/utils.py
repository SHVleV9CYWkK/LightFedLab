import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision import transforms
from torchvision.models import vgg16, resnet18, alexnet, resnet50
from torch import nn
import torch.optim as optim

from models.cnn_model import CNNModel, LeafCNN1, LeNet


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'emnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
        dataset = EMNIST(root='./data', train=True, download=True, transform=transform, split="byclass")
    elif dataset_name == 'mnist':
        transform = transforms.ToTensor()
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"dataset_name does not contain {dataset_name}")
    return dataset


def load_model(model_name, num_classes):
    if model_name == 'vgg16':
        model = vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'alexnet':
        model = alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'resnet18':
        model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'cnn':
        model = CNNModel(num_classes)
    elif model_name == 'leafcnn1':
        model = LeafCNN1(num_classes)
    elif model_name == 'lenet':
        model = LeNet(num_classes)
    else:
        raise ValueError(f"model_name does not contain {model_name}")
    return model


def get_client_data_indices(root_dir, dataset_name, split_method):
    dir_path = os.path.join(root_dir, f"{dataset_name}_{split_method}")

    if not os.path.exists(dir_path):
        raise ValueError(f"No matching dataset and split method found for {dataset_name} and {split_method}")

    # 获取客户端目录
    client_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    # 自动获取客户端数量
    num_clients = len(client_dirs)

    # 读取每个客户端的数据集索引
    client_indices = {}
    for client_dir in client_dirs:
        client_id = int(client_dir.split('_')[-1])
        client_indices[client_id] = {
            'train': os.path.join(dir_path, client_dir, 'train_indexes.npy'),
            'val': os.path.join(dir_path, client_dir, 'val_indexes.npy')
        }

    return client_indices, num_clients


def plot_training_results(base_path, result_path=None, metrics=None):
    if metrics is None:
        metrics = ['accuracy', 'f1', 'loss', 'precision', 'recall']

    if result_path is None:
        result_path = base_path.replace('logs', 'result_image')
    os.makedirs(result_path, exist_ok=True)  # 创建结果图片目录

    # 读取所有的方法目录
    methods = [dir for dir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, dir))]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # 遍历每个方法，并读取相应的度量文件
        for method in methods:
            metric_path = os.path.join(base_path, method, f'{metric}.txt')
            if os.path.exists(metric_path):
                data = pd.read_csv(metric_path, header=None)
                plt.plot(data, label=method)

        plt.title(f'Training {metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

        # 保存图像
        plt.savefig(os.path.join(result_path, f'{metric}.png'))
        plt.close()


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None, gated_learner=False):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        if gated_learner:
            # milestones = [n_rounds//2, 11*(n_rounds//12)]
            milestones = [3 * (n_rounds // 4)]
        else:
            milestones = [n_rounds // 2, 3 * (n_rounds // 4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif "reduce_on_plateau" in scheduler_name:
        last_word = scheduler_name.split("_")[-1]
        patience = int(last_word) if last_word.isdigit() else 10
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=0.75)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")
