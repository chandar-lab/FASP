#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=23:57:00
#SBATCH -o ./slurm-%j.out  # Write the log in $SCRATCH

python main.py --dataset Jigsaw --batch_size 64 --num_epochs 10 --classifier_model bert-base-cased 
