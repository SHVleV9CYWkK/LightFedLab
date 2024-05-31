import os
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST
from torchvision import transforms
from torchvision.models import vgg16, resnet18, alexnet
from torch import nn


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'mnist':
        dataset = EMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor(), split="digits")
    else:
        raise ValueError(f"dataset_name does not contain {dataset_name}")
    return dataset


def load_model(model_name, num_classes):
    if model_name == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'alexnet':
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'resnet18':
        model = resnet18(retrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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
