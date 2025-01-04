root=/home/zikang/temp/run/Grokking_tasks/logs

python main.py --training_fraction 0.5 --prime 97 --max_steps 10000 \
    --dropout 0.1 --noise_ratio 0.001 \
    --architecture LSTM \
    --log_npz $root/q2/lstm.npz --device cuda:3
python visualize.py --log_npz $root/q2/lstm.npz

python main.py --training_fraction 0.5 --prime 97 --max_steps 10000 \
    --dropout 0.1 --noise_ratio 0.001 \
    --architecture GNN \
    --log_npz $root/q2/gnn.npz --device cuda:3
python visualize.py --log_npz $root/q2/gnn.npz
