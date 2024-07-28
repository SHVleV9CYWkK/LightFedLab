# emnist experiments
#python build_statistical_heterogeneity.py --dataset_name emnist --clients_num 100 --split_method clusters --seed 42 --alpha 0.4 --frac 0.8 --dataset_indexes_dir client_indices
#python main.py --fl_method pfedgate --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.1 --gating_lr 0.1 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd --n_job 4
#python main.py --fl_method fedwcp --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.1 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd --n_job 4
#python main.py --fl_method fedavg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
#python main.py --fl_method fedmask --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
#python main.py --fl_method fedcg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.001 --server_lr 0.001 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
#python main.py --fl_method qfedcg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.001 --server_lr 0.01 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4

# 表现没有上面的好
#python main.py --fl_method pfedgate --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd
# -> 一致 python main.py --fl_method fedavg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd
# python main.py --fl_method fedmask --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.1 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam
# python main.py --fl_method fedcg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.01 --server_lr 0.01 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam

#cifar100 experiments 还有测试
# python build_statistical_heterogeneity.py --dataset_name cifar100 --clients_num 50 --split_method clusters --seed 42 --alpha 0.4 --frac 1 --dataset_indexes_dir client_indices
# python main.py --fl_method pfedgate --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.1 --gating_lr 1 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name sgd --n_job 4
# python main.py --fl_method fedwcp --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.001 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name adam--n_job 4
# python main.py --fl_method fedavg --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
# python main.py --fl_method fedmask --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.05 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
# python main.py --fl_method fedcg --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.001 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4
# python main.py --fl_method qfedcg --dataset_name cifar100 --model lenet --local_epochs 1 --lr 0.001 --batch_size 128 --n_rounds 200 --scheduler_name multi_step --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 4


# 记录emnist 最终脚本
# python main.py --fl_method pfedgate --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.1 --gating_lr 0.1 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name sgd --n_job 10
# python main.py --fl_method fedavg --dataset_name emnist --model leafcnn1 --local_epochs 1 --lr 0.005 --batch_size 128 --n_rounds 20 --seed 42 --device cuda --split_method clusters --optimizer_name adam --n_job 10