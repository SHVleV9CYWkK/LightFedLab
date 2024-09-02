import subprocess
import re
from itertools import product


def run_and_capture(params):
    # 打印当前参数配置
    print(f"Running with parameters: {params}")

    # 构建命令行参数
    cmd = ["python", "main.py"]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    # 运行命令并实时捕获输出
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    # 打印子进程的输出
    print(stdout)
    if stderr:
        print(stderr)

    # 正则表达式匹配准确率
    accuracy_pattern = r"Accuracy: ([0-9.]+)"
    accuracies = re.findall(accuracy_pattern, stdout)
    max_accuracy = max(map(float, accuracies)) if accuracies else None

    # 返回最大准确率
    return max_accuracy


if __name__ == '__main__':
    # 参数空间
    param_grid = {
        "fl_method": ["adfedwcp"],
        "dataset_name": ["cifar100"],
        "model": ["resnet18"],
        "local_epochs": [1],
        "lr": [0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [32],
        "n_rounds": [50],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["adam"],
        "n_job": [0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # 记录每个参数组合的最大准确率
    results = []

    # 遍历所有参数组合，执行并记录最大准确率
    for params in param_combinations:
        max_accuracy = run_and_capture(params)
        results.append((params, max_accuracy))
        with open("grid_search_results.txt", "a") as file:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")

    # 参数空间
    param_grid = {
        "fl_method": ["fedwcp"],
        "dataset_name": ["cifar100"],
        "model": ["resnet18"],
        "local_epochs": [1],
        "lr": [0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [32],
        "n_clusters": [8, 16, 32],
        "n_rounds": [50],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["adam"],
        "n_job": [0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # 记录每个参数组合的最大准确率
    results = []

    # 遍历所有参数组合，执行并记录最大准确率
    for params in param_combinations:
        max_accuracy = run_and_capture(params)
        results.append((params, max_accuracy))
        with open("grid_search_results.txt", "a") as file:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")

    # 参数空间
    param_grid = {
        "fl_method": ["fedavg", "fedmask", "qfedcg"],
        "dataset_name": ["cifar100"],
        "model": ["resnet18"],
        "local_epochs": [1],
        "lr": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [32],
        "n_rounds": [50],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["adam", "sgd"],
        "n_job": [0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # 记录每个参数组合的最大准确率
    results = []

    # 遍历所有参数组合，执行并记录最大准确率
    for params in param_combinations:
        max_accuracy = run_and_capture(params)
        results.append((params, max_accuracy))
        with open("grid_search_results.txt", "a") as file:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")

    # 参数空间
    param_grid = {
        "fl_method": ["fedem"],
        "dataset_name": ["cifar100"],
        "model": ["resnet18"],
        "local_epochs": [1],
        "lr": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "batch_size": [32],
        "n_rounds": [50],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["adam", "sgd"],
        "n_job": [0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # 记录每个参数组合的最大准确率
    results = []

    # 遍历所有参数组合，执行并记录最大准确率
    for params in param_combinations:
        max_accuracy = run_and_capture(params)
        results.append((params, max_accuracy))
        with open("grid_search_results.txt", "a") as file:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")

    # 参数空间
    param_grid = {
        "fl_method": ["pfedgate"],
        "dataset_name": ["cifar100"],
        "model": ["resnet18"],
        "local_epochs": [1],
        "lr": [0.1, 0.05, 0.01, 0.005, 0.001],
        "gating_lr": [2, 1, 0.5, 0.1, 0.05],
        "batch_size": [32],
        "n_rounds": [50],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["sgd"],
        "n_job": [0]
    }

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # 记录每个参数组合的最大准确率
    results = []

    # 遍历所有参数组合，执行并记录最大准确率
    for params in param_combinations:
        max_accuracy = run_and_capture(params)
        results.append((params, max_accuracy))
        with open("grid_search_results.txt", "a") as file:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")
