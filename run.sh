# python build_statistical_heterogeneity.py --dataset_name emnist --clients_num 100 --split_method clusters --seed 42 --alpha 0.4 --frac 0.8 --dataset_indexes_dir client_indices
python main.py --fl_method pfedgate --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.1 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd
python main.py --fl_method fedwcp --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam
python main.py --fl_method fedavg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam
python main.py --fl_method fedmask --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam
python main.py --fl_method fedcg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.001 --server_lr 0.001 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam
python main.py --fl_method qfedcg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.001 --server_lr 0.001 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam



# python build_statistical_heterogeneity.py --dataset_name emnist --clients_num 100 --split_method clusters --seed 42 --alpha 0.4 --frac 0.8 --dataset_indexes_dir client_indices
# python main.py --fl_method fedwcp --dataset_name emnist --model leafcnn1 --lr 1e-3 --batch_size 128 --n_rounds 30 --seed 42 --device mps --split_method clusters --is_send_gradients False]
# python main.py --fl_method pfedgate --dataset_name emnist --model leafcnn1 --lr 1e-3 --batch_size 128 --n_rounds 30 --seed 42 --device mps --split_method clusters --is_send_gradients False]
# python main.py --fl_method fedavg --dataset_name emnist --model leafcnn1 --lr 1e-3 --batch_size 128 --n_rounds 30 --seed 42 --device mps --split_method clusters --is_send_gradients False]
# python main.py --fl_method qfedcg --dataset_name emnist --model leafcnn1 --lr 1e-3 --batch_size 128 --n_rounds 30 --seed 42 --device mps --split_method clusters --is_send_gradients True]
# python main.py --fl_method fedcg - -dataset_name emnist --model leafcnn1 --lr 1e-3 --batch_size 128 --n_rounds 30 --seed 42 --device mps --split_method clusters --is_send_gradients True]
