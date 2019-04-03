#!/bin/bash
#SBATCH -A kaneki993
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1:00:00
#SBATCH --mincpus=12
#SBATCH --reservation=cvit-trial
module add cuda/9.0
module add cudnn/7-cuda-9.0


python3.5 train.py



# To run: sbatch magic.sh
