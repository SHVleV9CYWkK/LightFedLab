python main.py --fl_method fedwcp --dataset_name cifar10 --model alexnet --lr 1e-5 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients False
python main.py --fl_method fedavg --dataset_name cifar10 --model alexnet --lr 1e-5 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients False
#python main.py --fl_method fedcg  --dataset_name cifar10 --model alexnet --lr 1e-7 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients True
#python main.py --fl_method qfedcg --dataset_name cifar10 --model alexnet --lr 1e-7 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients True
python main.py --fl_method fedwcp --dataset_name mnist --model cnn --lr 1e-4 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients False
python main.py --fl_method fedavg --dataset_name mnist --model cnn --lr 1e-4 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients False
python main.py --fl_method fedcg --dataset_name mnist --model cnn --lr 1e-4 --server_lr 0.005 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients True
python main.py --fl_method qfedcg --dataset_name mnist --model cnn --lr 1e-4 --server_lr 0.005 --batch_size 16 --n_rounds 10 --seed 42 --device mps --split_method dirichlet --is_send_gradients True

