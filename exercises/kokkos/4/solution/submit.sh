#!/bin/bash
#

#SBATCH --job-name=submit
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu

# Load the required modules
module load gcc cuda kokkos


export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=10

./04_Exercise.Any
