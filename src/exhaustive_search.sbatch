#!/bin/bash

#SBATCH --job-name=exhaustive_search
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00

echo "SLURM_JOB_ID: $SLURM_JOB_ID."
echo "Executing on the machine:" $(hostname)

MAP=efficient_net_1
# MAP=efficient_net_5
# MAP=dist_yolo

module purge
module use $HOME/modulefiles
module add julia/1.10.0

echo "Start running Julia: " $(date)
SECONDS=0

(set -x; julia --project --threads=1 exhaustive_search.jl $MAP)

echo "End running Julia: " $(date)
echo "Elapsed time: $SECONDS"
