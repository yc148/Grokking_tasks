import argparse
from pathlib import Path
import numpy as np
import torch

from task_1.dataprocess import set_seed
from task_1.model_training import transformer_fraction

def _parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument("--training_fraction", type=float, required=True)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    # Regularization
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--optimizer_tag", type=str, default="AdamW")
    # Experiment
    parser.add_argument("--log_npz", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args

def _get_optimizer_configs(tag: str):
    if tag == "AdamW":
        type = torch.optim.AdamW
        cfg = {
            "lr": 1e-3,
            "betas": (0.9, 0.98),
            "weight_decay": 1.0,
        }
    else:
        raise NotImplementedError
    return type, cfg

def main(args: argparse.Namespace):
    set_seed(args.seed)
    optimizer_type, optimizer_cfg = _get_optimizer_configs(args.optimizer_tag)
    logs = transformer_fraction(
        fraction=args.training_fraction,
        modulus=args.prime,
        num_sum=2,
        max_opt_times=args.max_steps,
        dropout=args.dropout,
        noise_ratio=0,
        optim_class=optimizer_type,
        optim_params=optimizer_cfg,
    )
    logs = {tag: np.asarray(vlist, dtype=np.float32) for tag, vlist in logs.items()}
    
    log_path = Path(args.log_npz)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(log_path.as_posix(), **logs)

if __name__ == "__main__":
    args = _parse_args()
    main(args)
