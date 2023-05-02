#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --gres=gpu:1

./a.out

