#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J attack
#SBATCH -p phan
#SBATCH --output=results/logs/attack_clen.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
module load python
conda activate torch
python main.py --seed 1 --performance_metric auc --eval_round 20 --attack_round 100 --rounds 2000 --lr 0.1 --client_bs 128