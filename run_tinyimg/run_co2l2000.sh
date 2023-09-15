# Co2L
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --notes co2l2000_1 --batch_size 256 --model resnet18 --dataset tiny-imagenet --mem_size 2000 --epochs 50 --start_epoch 100 --learning_rate 0.1 --temp 0.5 --current_temp 0.1 --past_temp 0.1  --cosine
CUDA_VISIBLE_DEVICES=0 python main_linear_buffer.py --seed 0 --notes co2l2000_1 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_1_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_1_cosine_warm/
CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 0 --notes co2l2000_1 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_1_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_1_cosine_warm/

CUDA_VISIBLE_DEVICES=0 python main.py --seed 1 --notes co2l2000_2 --batch_size 256 --model resnet18 --dataset tiny-imagenet --mem_size 2000 --epochs 50 --start_epoch 100 --learning_rate 0.1 --temp 0.5 --current_temp 0.1 --past_temp 0.1  --cosine
CUDA_VISIBLE_DEVICES=0 python main_linear_buffer.py --seed 1 --notes co2l2000_2 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_2_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_2_cosine_warm/
CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 1 --notes co2l2000_2 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_2_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_2_cosine_warm/

CUDA_VISIBLE_DEVICES=0 python main.py --seed 2 --notes co2l2000_3 --batch_size 256 --model resnet18 --dataset tiny-imagenet --mem_size 2000 --epochs 50 --start_epoch 100 --learning_rate 0.1 --temp 0.5 --current_temp 0.1 --past_temp 0.1  --cosine
CUDA_VISIBLE_DEVICES=0 python main_linear_buffer.py --seed 2 --notes co2l2000_3 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_3_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_3_cosine_warm/
CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 2 --notes co2l2000_3 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_3_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_3_cosine_warm/

CUDA_VISIBLE_DEVICES=0 python main.py --seed 3 --notes co2l2000_4 --batch_size 256 --model resnet18 --dataset tiny-imagenet --mem_size 2000 --epochs 50 --start_epoch 100 --learning_rate 0.1 --temp 0.5 --current_temp 0.1 --past_temp 0.1  --cosine
CUDA_VISIBLE_DEVICES=0 python main_linear_buffer.py --seed 3 --notes co2l2000_4 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_4_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_4_cosine_warm/
CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 3 --notes co2l2000_4 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_4_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_4_cosine_warm/

CUDA_VISIBLE_DEVICES=0 python main.py --seed 4 --notes co2l2000_5 --batch_size 256 --model resnet18 --dataset tiny-imagenet --mem_size 2000 --epochs 50 --start_epoch 100 --learning_rate 0.1 --temp 0.5 --current_temp 0.1 --past_temp 0.1  --cosine
CUDA_VISIBLE_DEVICES=0 python main_linear_buffer.py --seed 4 --notes co2l2000_5 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_5_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_5_cosine_warm/
CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 4 --notes co2l2000_5 --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt ./save_random_2000/tiny-imagenet_models/co2l2000_5_cosine_warm/ --logpt ./save_random_2000/logs/tiny-imagenet/co2l2000_5_cosine_warm/