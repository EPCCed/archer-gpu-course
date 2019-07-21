#!/bin/bash
#
# cuda-memory
#
# Join stdout and sterr in submit.o{job_id}
# Set the queue and the resources
#
#PBS -N submit
#PBS -j oe
#PBS -q gpu-teach
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=0:01:00

# Budget: use either your default or the reservation
#PBS -A y15

# Load the required modules
module load gcc
module load cuda

# Pick a random device as PBS on Cirrus not yet configured to control
# GPU visibility
r=$RANDOM; let "r %= 4";
export CUDA_VISIBLE_DEVICES=$r
echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

cd $PBS_O_WORKDIR

./reverse
