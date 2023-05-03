#!/bin/bash

#SBATCH --time=00:02:00
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:2

module load nvidia/nvhpc
export TMPDIR=$(pwd)

./a.out
