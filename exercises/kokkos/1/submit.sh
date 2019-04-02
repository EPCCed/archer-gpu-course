#!/bin/bash
#
#PBS -N submit
#PBS -j oe
#PBS -q gpu-teach
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=0:01:00

# Budget: use either your default or the reservation
#PBS -A y15

# Load the required modules
module load gcc cuda kokkos

cd $PBS_O_WORKDIR

export OMP_PLACES=cores
export OMP_PROC_BIND=close

for threads in 1 2 4 8; do
    export OMP_NUM_THREADS=$threads
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    ./01_Exercise.OpenMP
done
