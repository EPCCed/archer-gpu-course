#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --partition=gpu-skylake
#SBATCH --qos=short
#SBATCH --gres=gpu:1


echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

./intro
