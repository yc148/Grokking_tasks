root=/home/zikang/temp/run/Grokking_tasks/logs

fraction_list=(
    0.3 \
    0.5 \
    0.7
)
for i in "${!fraction_list[@]}"; do
    fraction=${fraction_list[$i]}
    python main.py --training_fraction $fraction --prime 97 --max_steps 10000 \
        --dropout 0.1 --noise_ratio 0.001 \
        --log_npz $root/q1/$fraction.npz --device cuda:3
    python visualize.py --log_npz $root/q1/$fraction.npz
done