#!/bin/bash
#SBATCH --job-name=kokkos-4
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=10

./04_Exercise.Any
