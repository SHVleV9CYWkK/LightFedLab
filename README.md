# FedWCP
Command:
python build_statistical_heterogeneity.py --dataset_name mnist --clients_num 10 --split_method dirichlet --seed 42 --alpha 0.1 --dataset_indexes_dir client_indices

python main.py --fl_method fedcg --dataset_name mnist --model cnn --lr 1e-4 --server_lr 0.0005 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients True
