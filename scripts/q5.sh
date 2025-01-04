root=/home/zikang/temp/run/Grokking_tasks/logs

log_npz=$root/q5/2.npz
python main.py --training_fraction 0.5 --prime 97 --max_steps 3000 \
    --dropout 0.1 --noise_ratio 0.001 --max_grad_norm 1.0 \
    --perturb_ratio 0.1 \
    --log_npz $log_npz --device cuda:3
python visualize.py --task q5 --log_npz $log_npz
