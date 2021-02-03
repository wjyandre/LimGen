## this shell file can run on slurm for gpu computing
##--constraint=v100, recommended v100, compsci-gpu line need to change to specific node name.

#!/bin/bash
#SBATCH --job-name=MTfemale
#SBATCH -t 48:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=200G --cpus-per-task=20
#SBATCH -p compsci-gpu 
#SBATCH -a 0-100 --ntasks-per-node=1
srun python3 run_Limericks.py -sl "story" -dir "MTBS" -m "multi" -ser 12 -re 30 -g "female" -bs "MTBS" -mas True
