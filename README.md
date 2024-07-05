
# LightFedLab

This repository hosts a lightweight federated learning (FL) framework designed for academic purposes. It focuses on simplifying the experimental process of FL algorithms by providing clear and simple base classes (abstract classes). This approach eliminates the need for industry-specific code, such as Docker, making it ideal for algorithm-focused research without concerning about specific real-world implementations.

## Features

- **Simplicity**: Easy-to-understand base classes for quick implementation and experimentation of various FL algorithms.
- **Lightweight**: No unnecessary industrial components; focus strictly on the algorithms.
- **Flexibility**: Designed to facilitate the exploration of novel federated learning algorithms.

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine along with necessary libraries which can be installed via:

```bash
pip install -r requirements.txt
```

### Generating Heterogeneous Datasets

The framework allows for the generation of heterogeneous datasets to simulate real-world client data distribution. Here’s how to generate a dataset for 10 clients using a Dirichlet distribution:

```bash
python build_statistical_heterogeneity.py --dataset_name mnist --clients_num 10 --split_method dirichlet --seed 42 --alpha 0.1 --dataset_indexes_dir client_indices
```

This command will create `data` and `client_indices` directories in the root directory, where the original datasets and client data distribution records are stored.

### Running Federated Learning Experiments

To run a federated learning experiment using the FedAvg algorithm with the following settings:

```bash
python main.py --fl_method fedavg --dataset_name mnist --model cnn --lr 1e-4 --server_lr 0.0005 --batch_size 16 --n_rounds 10 --seed 42 --device cpu --split_method dirichlet --is_send_gradients True
```

### Parameters

- `--fl_method`: The federated learning method to use (e.g., `fedavg`).
- `--dataset_name`: Name of the dataset (e.g., `mnist`).
- `--model`: Model to train (e.g., `cnn`).
- `--lr`: Learning rate for the client model.
- `--server_lr`: Server-side learning rate.
- `--batch_size`: Batch size for training.
- `--n_rounds`: Number of federated rounds to execute.
- `--seed`: Seed for random number generation.
- `--device`: Device to run the training on (`cpu`, `gpu`, `mps`).
- `--split_method`: Method used to split the dataset (`dirichlet`).
- `--is_send_gradients`: Whether to send gradients back to server.

## Contributing

We welcome contributions to improve the framework! Please submit pull requests or open an issue to discuss your ideas.

## License

Distributed under the MIT License. See `LICENSE` for more information.
