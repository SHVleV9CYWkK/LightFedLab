import os
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision import transforms
from torchvision.models import vgg16, resnet18, alexnet, resnet50
from torch import nn
from models import CNNModel


def load_dataset(dataset_name, model_name):

    transform_list = [transforms.ToTensor()]

    if model_name in ['vgg16', 'resnet18', 'resnet50', 'alexnet']:
        transform_list.insert(0, transforms.Resize(224))
    elif model_name == 'cnn':
        transform_list.insert(0, transforms.Resize(28))

    transform = transforms.Compose(transform_list)

    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'emnist':
        dataset = EMNIST(root='./data', train=True, download=True, transform=transform, split="digits")
    elif dataset_name == 'mnist':
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
