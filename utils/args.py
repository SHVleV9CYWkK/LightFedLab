import argparse


def parse_args_for_dataset():
    parser = argparse.ArgumentParser(description="Dataset splitting for federated learning")
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'emnist'],
                        help='dataset name')
    parser.add_argument('--clients_num', type=int, default=10, help='number of clients')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--split_method', type=str, default='train', choices=['dirichlet', 'label'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Parameters that control the degree of non-IID.'
             'The smaller the alpha, the greater the task difference with dirichlet split method',
    )
    parser.add_argument(
        '--number_label', type=int, default=2,
        help='Parameters that control the degree of non-IID.'
             'Controls the number of label types owned by the local client with label split method',
    )
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    args = parser.parse_args()
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl_method', type=str, default='fedavg', choices=['fedavg', 'qfedcg'],
                        help='federated learning method')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--model', type=str, default='vgg16', choices=['vgg16', 'resnet9', 'resnet18'],
                        help='model name')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--local_epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_rounds', type=int, default=1, help='number of global rounds')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='determine the computing platform')
    parser.add_argument('--split_method', type=str, default='train', choices=['dirichlet', 'label'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    args = parser.parse_args()
    return args
