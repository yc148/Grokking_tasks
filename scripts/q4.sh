root=/home/zikang/temp/run/Grokking_tasks/logs

p_list=(
    23 \
    23 \
    23
)
num_sum_list=(
    2 \
    3 \
    4
)
for i in "${!p_list[@]}"; do
    p=${p_list[$i]}
    num_sum=${num_sum_list[$i]}
    python main.py --training_fraction 0.3 --prime $p --max_steps 10000 --num_sum $num_sum \
        --dropout 0.1 --noise_ratio 0.001 \
        --log_npz $root/q4/$p-$num_sum.npz --device cuda:3
    python visualize.py --log_npz $root/q4/$p-$num_sum.npz
done
