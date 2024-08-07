import argparse


def parse_args_for_dataset():
    parser = argparse.ArgumentParser(description="Dataset splitting for federated learning")
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'yahooanswers'],
                        help='dataset name')
    parser.add_argument('--clients_num', type=int, default=10, help='number of clients')
    parser.add_argument('--n_clusters', type=int, default=-1, help='number of clusters using clusters split method')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--split_method', type=str, default='train', choices=['dirichlet', 'label', 'clusters'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Parameters that control the degree of non-IID.'
             'The smaller the alpha, the greater the task difference',
    )

    parser.add_argument(
        '--frac', type=float, default=1.0,
        help='The proportion of a partial dataset to the entire dataset is adopted')

    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        help='The proportion of the test set to the overall dataset')

    parser.add_argument(
        '--number_label', type=int, default=2,
        help='Parameters that control the degree of non-IID.'
             'Controls the number of label types owned by the local client with label split method',
    )
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    args = parser.parse_args()
    return args


def parse_args_for_visualization():
    parser = argparse.ArgumentParser(description="Visualize the parameters of the training process")
    parser.add_argument('--log_dir', type=str, required=True, help='log directory')
    parser.add_argument('--save_dir', type=str, default=None, help='save directory')
    args = parser.parse_args()
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl_method', type=str, default='fedavg', choices=['fedavg', 'fedcg', 'qfedcg', 'fedwcp', 'pfedgate', 'fedmask', 'fedem'],
                        help='federated learning method')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'emnist', 'mnist', 'yahooanswers'],
                        help='dataset name')
    parser.add_argument('--model', type=str, default='alexnet', choices=['vgg16', 'resnet18', 'cnn', 'resnet50', 'alexnet', 'leafcnn1', 'lenet', 'mobilebart'],
                        help='model name')
    parser.add_argument('--optimizer_name', type=str, default='adam', choices=['sgd', 'adam', 'adamw'],
                        help='The name of the optimizer used')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate of the local client during training')
    parser.add_argument('--server_lr', type=float, default=1e-3, help='When aggregating global gradients, the learning rate when the global model is updated')
    parser.add_argument('--gating_lr', type=float, default=1, help='pFedgate\'s learning rate at the gate layer')
    parser.add_argument('--client_selection_rate', type=float, default=1, help='Client sampling rate')
    parser.add_argument('--local_epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_rounds', type=int, default=1, help='number of global rounds')
    parser.add_argument('--scheduler_name', type=str, default='reduce_on_plateau', choices=['sqrt', 'linear', 'constant', 'cosine_annealing', 'multi_step', 'reduce_on_plateau'],
                        help='Select the name of the learning rate scheduler')
    parser.add_argument('--sparse_factor', type=float, default=0.5, help='Set the sparsity for the method that requires the sparsity or compression ratio')
    parser.add_argument('--num_components', type=int, default=2, help='Number of components for FedEM')
    parser.add_argument('--quantization_levels', type=int, default=1, help='The default quantization level for QFedCG')
    parser.add_argument('--is_send_gradients', type=bool, default=False, help='Controls whether the client uploads gradient aggregations, FedCG and QFedCG are not controlled by this parameter.')
    parser.add_argument('--n_job', type=int, default=1, help='The number of processes that execute client training in parallel in the server')
    parser.add_argument('--dl_n_job', type=int, default=0, help='The number of parallels in the client\'s Dataload')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='log directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='determine the computing platform')
    parser.add_argument('--split_method', type=str, choices=['dirichlet', 'label', 'clusters'],
                        help='The methods of splitting the data set to generate non-IID are dirichlet and label '
                             'respectively. dirichlet is using dirichlet distributed. label indicates that the client '
                             'owns a subset of label')
    parser.add_argument('--dataset_indexes_dir', type=str, default='client_indices',
                        help='The root directory of the local client dataset index')

    parser.add_argument('--enable_scheduler', type=bool, default=True, help='whether to enable the learning rate scheduler')

    args = parser.parse_args()
    return args
