#!/bin/bash
#
# Join stdout and sterr in submit.o{job_id}
# Set the queue and the resources
#

#SBATCH --time=00:20:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# Load the required modules

echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

./reconstruct
