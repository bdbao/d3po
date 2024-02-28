#!/bin/bash
#SBATCH --job-name=install
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --output=log_install.out
#SBATCH --error=log_install.err
#SBATCH --nodelist=gpu01
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-user=baobuiduy.vn@gmail.com

#export TMPDIR='/var/tmp'
#conda deactivate
#conda create --name d3po python=3.10
#conda activate d3po

TMPDIR=/dev/sda2 pip install --cache-dir=$TMPDIR ml-collections absl-py diffusers==0.17.1 wandb torchvision inflect==6.0.4 pydantic==1.10.9 transformers==4.30.2 accelerate==0.22.0 torch==2.0.1
# pip install ml-collections absl-py diffusers==0.17.1 wandb torchvision inflect==6.0.4 pydantic==1.10.9 transformers==4.30.2 accelerate==0.22.0 torch==2.0.1
