#!/bin/bash
#SBATCH -A project02524              # Replace with your project/account name
#SBATCH --partition=gpu
#SBATCH -J TUSZ_train_tusz           # Job name
#SBATCH -n 1                         # Number of tasks
#SBATCH -c 4                         # Number of CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --gpus-per-task=1            # Assign 1 GPU to each task
#SBATCH --mem-per-cpu=8000           # 4 * 32 GB = 128 GB total
#SBATCH --time=8:00:00               # Maximum runtime (adjust as needed)
#SBATCH -o logs/train_dual_out_%j.log
#SBATCH -e logs/train_dual_err_%j.log

module load cuda
module load gcc/8 python/3.10


# Use your custom Python installation
cd /work/home/nj31voho/thesis/eeg/models/

# Each task activates its own virtual environment
srun --exclusive -N1 -n1 bash -c 'source ./venv/bin/activate && which python && python -c "import torch; print(torch.__version__)" && python visualize_preprocess.py' > logs/corrupt_dataset_%j.out 2>&1 &

wait
