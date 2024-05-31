import torch
from utils.args import *
from utils.utils import *
from client import ClientFactory


# def _save_log(self, eval_results):
#     # 获取今天的日期字符串
#     today_date = datetime.today().strftime('%Y-%m-%d')
#
#     # 创建今天日期的目录
#     log_dir = os.path.join(self.save_log_dir, today_date)
#     os.makedirs(log_dir, exist_ok=True)
#
#     # 遍历评估结果，保存到相应的文件中
#     for metric, value in eval_results.items():
#         file_path = os.path.join(log_dir, f"{metric}.txt")
#         with open(file_path, 'a') as file:
#             file.write(f"{value}\n")
#
#     print(f"Evaluation results saved to {log_dir}")

if __name__ == '__main__':
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    full_dataset = load_dataset(args.dataset_name)
    if args.dataset_name == 'cifar10' or args.dataset_name == 'emnist':
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    model = load_model(args.model_name, num_classes=num_classes).to(device)

    client_indices, num_clients = get_client_data_indices(args.dataset_indexes_dir, args.dataset_name, args.split_method)

    criterion = torch.nn.CrossEntropyLoss()
    clients = ClientFactory().create_client(num_clients, args.fl_method,
                                            client_indices, full_dataset,
                                            args.batch_size, args.lr,
                                            args.local_epochs, criterion, device)



