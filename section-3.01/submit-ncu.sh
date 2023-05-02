#!/bin/bash

#SBATCH --time=00:02:00
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1

module load nvidia/nvhpc

# Temporary files may have to be stored in a user directory with
# sufficient space ...
export TMPDIR=$(pwd)

ncu -o default ./a.out

