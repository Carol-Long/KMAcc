#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-4:59
#SBATCH -p sapphire
#SBATCH --mem=128000
#SBATCH --mail-type=ALL

module load python/3.10.12-fasrc01
mamba activate torchy
python exp1.py
