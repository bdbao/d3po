#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --output=log_sample.out
#SBATCH --error=log_sample.err
#SBATCH --nodelist=gpu01
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-user=baobuiduy.vn@gmail.com

accelerate launch ../scripts/sample.py