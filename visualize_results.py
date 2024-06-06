from utils.utils import plot_training_results
from utils.args import parse_args_for_visualization

if __name__ == '__main__':
    arguments = parse_args_for_visualization()
    plot_training_results(arguments.log_dir, arguments.save_dir)