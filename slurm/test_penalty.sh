#!/bin/sh
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=rec_5000_1
#SBATCH --output=tmp/rec_5000_1.out
#SBATCH --error=tmp/rec_5000_1.err

module load gcc/12.2.0
module load cuda/12.1.1
module load cudnn/8.9.7.29-12
module load openblas/0.3.24
module load python/3.11.6

source venv/bin/activate
python3 ../tests/test_penalty.py
