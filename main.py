import os
import time
from datetime import datetime
import torch
from utils.args import parse_args
from utils.utils import load_model, load_dataset, get_client_data_indices
from client import ClientFactory
from server import ServerFactory


def save_log(eval_results, save_log_dir, fl_type):
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Create a directory for today's date
    today_dir = os.path.join(save_log_dir, today_date)
    os.makedirs(today_dir, exist_ok=True)

    # Create a directory for FL
    log_dir = os.path.join(today_dir, fl_type)
    os.makedirs(log_dir, exist_ok=True)

    for metric, value in eval_results.items():
        file_path = os.path.join(log_dir, f"{metric}.txt")
        with open(file_path, 'a') as file:
            file.write(f"{value}\n")

    print(f"Evaluation results saved to {log_dir}")


def execute_fed_process(server, args):
    for r in range(args.n_rounds):
        print(f"------------\nRound {r}")
        start_time = time.time()
        server.train()
        eval_results = server.evaluate()
        end_time = time.time()
        eval_results_str = ', '.join([f"{metric.capitalize()}: {value:.4f}" for metric, value in eval_results.items()])
        print(f"Training time: {end_time - start_time}. Evaluation Results: {eval_results_str}")
        print("Evaluation Results:")
        for metric, value in eval_results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        save_log(eval_results, args.log_dir, args.fl_method)


def execute_experiment(args, device):
    full_dataset = load_dataset(args.dataset_name)
    if args.dataset_name == 'cifar10' or args.dataset_name == 'emnist':
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    model = load_model(args.model, num_classes=num_classes)

    client_indices, num_clients = get_client_data_indices(args.dataset_indexes_dir, args.dataset_name,
                                                          args.split_method)

    criterion = torch.nn.CrossEntropyLoss()
    clients = ClientFactory().create_client(num_clients, args.fl_method,
                                            client_indices, full_dataset,
                                            args.batch_size, args.lr,
                                            args.local_epochs, criterion, device)

    central_server = ServerFactory().create_server(args.fl_method, clients, model, args.client_selection_rate, args.server_lr)

    execute_fed_process(central_server, args)


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.device == "cuda" and torch.cuda.is_available():
        compute_device = torch.device("cuda")
    elif arguments.device == "mps" and torch.backends.mps.is_available():
        compute_device = torch.device("mps:0")
    else:
        compute_device = torch.device("cpu")
    print(f"Using device: {compute_device}")

    execute_experiment(arguments, compute_device)
