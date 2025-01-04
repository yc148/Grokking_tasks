root=/home/zikang/temp/run/Grokking_tasks/logs

fraction_list=(
    0.3 \
    0.4 \
    0.5 \
    0.6 \
    0.7 \
    0.8 \
    0.9 \
    0.25 \
    0.35 \
    0.45 \
    0.55 \
    0.65 \
    0.75 \
    0.85
)

for i in "${!fraction_list[@]}"; do
    fraction=${fraction_list[$i]}
    python main.py --training_fraction $fraction --prime 97 --max_steps 100000 \
        --dropout 0 --noise_ratio 0.01 --optimizer_tag Adam \
        --log_npz $root/q3/adam_noise/$fraction.npz --device cuda:3
done
