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
        "fl_method": ["fedavg"],
        "dataset_name": ["emnist"],
        "model": ["leafcnn1"],
        "local_epochs": [1],
        "lr": [0.1, 0.05, 0.01, 0.005, 0.001],
        "batch_size": [128],
        "n_rounds": [20],
        "seed": [42],
        "device": ["cuda"],
        "split_method": ["clusters"],
        "optimizer_name": ["sgd", "adam"],
        "n_job": [10]
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

    # 将所有参数组合及其最大准确率保存到文本文件
    with open("grid_search_results.txt", "w") as file:
        for params, max_accuracy in results:
            file.write(f"Params: {params}, Max Accuracy: {max_accuracy}\n")
