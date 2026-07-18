#!/bin/bash
export PATH="$PATH:/usr/bin"

module load cuda/12.9
module load gcc/9
source /home/pselvaraju/miniforge3/etc/profile.d/conda.sh
conda activate Oktopus

python3 train_net_3dvec.py -c config/config_grid_b13d3_oktopus_boots1.yaml 
