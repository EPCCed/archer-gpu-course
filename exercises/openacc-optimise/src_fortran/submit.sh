#!/bin/bash
#
# Join stdout and sterr in submit.o{job_id}
# Set the queue and the resources
#

#SBATCH --job-name=submit
#SBATCH --time=00:03:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1


echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

./reconstruct
