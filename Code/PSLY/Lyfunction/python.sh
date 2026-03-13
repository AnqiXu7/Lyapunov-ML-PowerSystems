#!/bin/bash
#SBATCH --job-name=extract06
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=log/extract06_%j.out
#SBATCH --error=log/extract06_%j.err

cd /users/sgaxu2/Lyfunction
python 01_extract_2.py
