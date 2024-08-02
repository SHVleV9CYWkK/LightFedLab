import os
import matplotlib.pyplot as plt
import pandas as pd
from utils.args import parse_args_for_visualization


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


if __name__ == '__main__':
    arguments = parse_args_for_visualization()
    plot_training_results(arguments.log_dir, arguments.save_dir, ['accuracy', 'loss', 'f1'])

