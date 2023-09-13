# Co2L
# CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --notes co2l500_1 --batch_size 512 --model resnet18 --dataset cifar100 --mem_size 500 --epochs 100 --start_epoch 100 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --wandb_project Co2L

CUDA_VISIBLE_DEVICES=1 python main_linear_buffer.py --seed 0 --notes co2l500_1 --dataset cifar100 --learning_rate 1 --target_task 9  --wandb_project Co2L --ckpt ./save_random_500/cifar100_models/co2l500_1/ --logpt ./save_random_500/logs/co2l500_1/

# CUDA_VISIBLE_DEVICES=0 python main_linear_all.py --seed 0 --notes co2l500_1 --dataset cifar100 --learning_rate 1 --target_task 9  --wandb_project Co2L --ckpt ./save_random_500/cifar100_models/co2l500_1/ --logpt ./save_random_500/logs/co2l500_1/


