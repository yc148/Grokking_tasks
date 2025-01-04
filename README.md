# Grokking_tasks
This is the code for the final project of PKU MIML 24 fall.

## Dependency
Our code uses standard python libraries and does not require specific versions. 
Our experiment uses below versions and anaconda env:

- python=3.10.0
- matplotlib=3.10.0
- numpy=1.23.1
- torch=2.5.1
- torch_geometric=2.6.1
- tqdm=4.64.0

## How to run
We include the scripts to reproduce our results in the scripts folder. 
For instance, to run task 1, execute
```bash
bash scripts/q1.sh
``` 
in the main directory. 
Please modify the root variable in the sh file to assign the position of saved log file before the run.

## Code file content
Here is a short description of function of each file.
- main.py: command line entrance for running experiments.
- visualize.py: command line entrance for visualizing stored logs.
- src/dataprocess.py: generate dataset.
- src/model_training.py: training procedure.
- src/transformer.py: Transformer architecture.
