#!/usr/bin/bash
#
# Author: Zehua Wang
# Updated: November 8, 2022

# slurm options: --------------------------------------------------------------
#SBATCH --job-name=eval_sim_scores
#SBATCH --mail-user=wangzeh@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=2:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/SI650/SI650F22_FinalProject/logs/%x-%j.log
#SBATCH --error=/home/%u/SI650/SI650F22_FinalProject/logs/%x-%j-E.log

# application: ----------------------------------------------------------------

# modules
module load python/3.9.12

cd /home/wangzeh/SI650/SI650F22_FinalProject/
source .venv/bin/activate
# pip install -r requirements.txt
python evaluation.py