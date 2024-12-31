import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from seaborn import color_palette
from typing import Optional, Tuple, List

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_npz", type=str, required=True)
    parser.add_argument("--downsample", type=int, default=0)
    args = parser.parse_args()
    return args

def save_fig(png_file: str):
    file_path = Path(png_file)
    if file_path.suffix != ".png":
        file_path = Path(f"{file_path.stem}.png")
    if file_path.is_file():
        file_path.unlink()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_file, bbox_inches='tight', dpi=300)
    plt.close()

def plot_multiple_lines(
    png_file: str,
    x: np.ndarray,
    y_list: List[np.ndarray],
    colors: Optional[List] = None,
    label_list: Optional[List[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_ticks: Optional[np.ndarray] = None,
    y_ticks: Optional[np.ndarray] = None,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    horizonal_lines: bool = False,
    vertical_lines: bool = False,
    downsample: int = 0,
):
    if colors is None:
        colors = color_palette("husl", len(y_list))

    plt.figure(figsize=(6, 4))

    if downsample > 0:
        x = x[::downsample]
        y_list = [y[::downsample] for y in y_list]
    
    for i in range(len(y_list)):
        label = label_list[i] if label_list is not None else f"{i}"
        plt.plot(x, y_list[i], color=colors[i], label=label)

    if horizonal_lines and y_ticks is not None:
        for y in y_ticks:
            plt.axhline(y=y, color="lightgray", linestyle="-", linewidth=0.8)
    if vertical_lines and x_ticks is not None:
        for x in x_ticks:
            plt.axvline(x=x, color="lightgray", linestyle="-", linewidth=0.8)

    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.legend()
    
    save_fig(png_file)

def main(args: argparse.Namespace):
    log_npz = Path(args.log_npz)
    with np.load(log_npz.as_posix()) as data:
        logs = {key: data[key] for key in data}
    workdir = log_npz.parent.joinpath(log_npz.stem)
    workdir.mkdir(exist_ok=True)

    steps = logs["step"]
    log_steps = np.log10(steps + 1)
    for key in ["loss", "acc"]:
        for x, tag in zip([steps, log_steps], ["", "_log"]):
            plot_multiple_lines(
                workdir.joinpath(f"{key}{tag}.png"),
                x=x,
                y_list=[logs[f"train_{key}"], logs[f"val_{key}"]],
                label_list=[f"train_{key}", f"val_{key}"],
                downsample=args.downsample,
            )

if __name__ == "__main__":
    args = _parse_args()
    main(args)
