#!/bin/bash -l

# Use the current working directory
#SBATCH -D ./

# Use the current environment for this job
#SBATCH --export=ALL

# Define job name
#SBATCH -J train

# Define a standard output file
#SBATCH -o crypto_lstm.%N.%j.out

# Define a standard error file
#SBATCH -e crypto_lstm.%N.%j.err


# Request the GPU partition (gpu).
#SBATCH -p gpu
# Request the number of nodes
#SBATCH -N 1
# Request the number of GPUs
#SBATCH --gres=gpu:1
# Request the number of CPU cores
#SBATCH -n 6
# Set time limit in format a-bb:cc:dd
#SBATCH -t 1-00:00:00


# Set your maximum stack size to unlimited
ulimit -s unlimited

# Set OpenMP thread number
export OMP_NUM_THREADS=$SLURM_NTASKS

# ========== Load anaconda and relevant modules ================
module load apps/anaconda3/2020.02

# Activate the Python environment
source activate my_env  # Ensure you have the correct environment set up with required packages

# List all modules
module list

echo =========================================================
echo SLURM job: submitted date = `date`
date_start=`date +%s`

hostname
echo Current directory: `pwd`
echo "Running Crypto LSTM job:"

# Execute the Python program
python train.py  # Assuming your script is named `main.py`. Adjust if different.
# Deactivate the Python environment
source deactivate

date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))

echo =========================================================
echo SLURM job: finished date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================

