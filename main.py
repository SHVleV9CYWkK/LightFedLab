import os
import time
from datetime import datetime
import torch
from utils.args import parse_args
from utils.utils import load_model, load_dataset, get_client_data_indices
from clinets.client_factory import ClientFactory
from servers.server_factory import ServerFactory


def save_log(eval_results, save_log_dir, dataset_name, fl_type):
    today_date = datetime.today().strftime('%Y-%m-%d')

    today_dir = os.path.join(save_log_dir, today_date)
    os.makedirs(today_dir, exist_ok=True)

    dataset_dir = os.path.join(today_dir, dataset_name)
    os.makedirs(today_dir, exist_ok=True)

    log_dir = os.path.join(dataset_dir, fl_type)
    os.makedirs(log_dir, exist_ok=True)

    for metric, value in eval_results.items():
        file_path = os.path.join(log_dir, f"{metric}.txt")
        with open(file_path, 'a') as file:
            file.write(f"{value}\n")


def execute_fed_process(server, args):
    for r in range(args.n_rounds):
        print(f"------------\nRound {r}")
        start_time = time.time()
        server.train()
        eval_results = server.evaluate()
        end_time = time.time()
        eval_results_str = ', '.join([f"{metric.capitalize()}: {value:.4f}" for metric, value in eval_results.items()])
        print(f"Training time: {(end_time - start_time):.2f}. Evaluation Results: {eval_results_str}")
        save_log(eval_results, args.log_dir, args.dataset_name, args.fl_method)


def execute_experiment(args, device):
    full_dataset = load_dataset(args.dataset_name, args.model)
    if args.dataset_name == 'cifar10' or args.dataset_name == 'emnist' or args.dataset_name == 'mnist':
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    model = load_model(args.model, num_classes=num_classes).to(device)

    client_indices, num_clients = get_client_data_indices(args.dataset_indexes_dir, args.dataset_name,
                                                          args.split_method)

    criterion = torch.nn.CrossEntropyLoss()
    clients = ClientFactory().create_client(num_clients, args, client_indices, full_dataset, criterion, device)

    central_server = ServerFactory().create_server(args, clients, model, device)

    execute_fed_process(central_server, args)


if __name__ == '__main__':
    arguments = parse_args()
    torch.manual_seed(arguments.seed)

    if arguments.device == "cuda" and torch.cuda.is_available():
        compute_device = torch.device("cuda")
    elif arguments.device == "mps" and torch.backends.mps.is_available():
        compute_device = torch.device("mps:0")
    else:
        compute_device = torch.device("cpu")
    print(f"Using device: {compute_device}")

    execute_experiment(arguments, compute_device)
    print("Done")


