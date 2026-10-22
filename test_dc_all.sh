#!/bin/bash
export PATH="$PATH:/usr/bin"

module load cuda/12.4
module load gcc/9
source /home/pselvaraju/miniforge3/etc/profile.d/conda.sh
conda activate Oktopus


#python test_dc.py     -c config/config_grid_b13d3_oktopus_puffer.yaml     -o test_adapt_oktopus_puffer_t1_dcsdd_r512     -s oktopus_9_v1     -y adapt_oktopus_puffer_t1.yaml     -r 512     -ckpt final     --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd     --dcsdd-methods dc,ours     --dcsdd-outer-iters 10 --dcsdd-mu 0.5   --dcsdd-dc-weight 0.4 --dcsdd-inner-iters 10     --dcsdd-verbose



python test_dc.py \
    -c config/config_grid_b13d3_oktopus_puffer.yaml \
    -s oktopus_9_v1 \
    -y adapt_oktopus_puffer_full_0067_sdf_safe.yaml \
    -o adapt_oktopus_puffer_optim_r1024 \
    -r 1024 \
    -ckpt final \
    -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 \
    --dcsdd-inner-iters 3 \
    --dcsdd-mu 0.5 \
    --dcsdd-dc-weight 0.4 \
    --dcsdd-verbose \
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-crop-grid \
    --dcsdd-fast-bbox-marking \
    --dcsdd-bounded-filtering \
    --dcsdd-filter-batch-size 262144 \
    --sdf-localize-after-gate \
    --surface-extraction ours



python test_dc.py \
    -c config/config_grid_b13d3_oktopus_boots1.yaml \
    -o test_adapt_oktopus_boots_t7_dcsdd_r1024_crop \
    -s oktopus_9_v1 \
    -y adapt_oktopus_boots_t7_boot1_only.yaml \
    -r 1024 \
    -ckpt final \
    -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 \
    --dcsdd-inner-iters 3\
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-mu 0.5   \
    --dcsdd-dc-weight 0.4 \
    --dcsdd-verbose \
    --dcsdd-bounded-filtering \
    --dcsdd-filter-batch-size 262144 \
    --dcsdd-compact-staging \
    --dcsdd-crop-grid \
    --dcsdd-fast-bbox-marking \
    --dcsdd-stream-inference \
    --dcsdd-stream-crop-first \
    --dcsdd-stream-replay-metadata \
    --dcsdd-stream-one-pass \
    --sdf-localize-after-gate \
    --dcsdd-stream-replay-dir /fast/pselvaraju/Oktopus_now/logs \
    --surface-extraction ours 

    #--dcsdd-stream-accepted-count 376525454 \

python test_dc.py \
    -c config/config_grid_b13d3_oktopus_cone.yaml \
    -o test_adapt_oktopus_boots_t7_dcsdd_r512_crop \
    -s oktopus_9_v1 \
    -y adapt_oktopus_cone_optimization.yaml \
    -r 512 \
    -ckpt final \
    -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 \
    --dcsdd-inner-iters 3\
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-mu 0.5   \
    --dcsdd-dc-weight 0.4 \
    --dcsdd-verbose \
    --dcsdd-bounded-filtering \
    --dcsdd-filter-batch-size 262144 \
    --dcsdd-compact-staging \
    --dcsdd-crop-grid \
    --dcsdd-fast-bbox-marking \
    --dcsdd-stream-inference \
    --sdf-localize-after-gate \
    --dcsdd-stream-replay-dir /fast/pselvaraju/Oktopus_now/logs \
    --surface-extraction ours 

python test_dc.py \
    -c config/config_grid_b13d3_oktopus_cone.yaml \
    -o test_adapt_oktopus_cone_dcsdd_r512 \
    -s oktopus_9_v1 \
    -y adapt_oktopus_cone_0045_fast.yaml \
    -r 512 \
    -ckpt final \
    -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 \
    --dcsdd-inner-iters 3 \
    --dcsdd-mu 0.5 \
    --dcsdd-dc-weight 0.4 \
    --dcsdd-verbose \
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-crop-grid \
    --dcsdd-fast-bbox-marking \
    --sdf-localize-after-gate \
    --sdf-optimize-zero-angular-wrap \
    --surface-extraction ours


python test_dc.py \
    -c config/config_grid_b13d3_oktopus_cone.yaml \
    -o test_adapt_oktopus_cone_dcsdd_r512 \
    -s oktopus_9_v1 \
    -y adapt_oktopus_cone_0048_interval_audit.yaml \
    -r 512 \
    -ckpt final \
    -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 \
    --dcsdd-inner-iters 3 \
    --dcsdd-mu 0.5 \
    --dcsdd-dc-weight 0.4 \
    --dcsdd-verbose \
    --dcsdd-crop-grid \
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-stream-replay-dir /fast/pselvaraju/Oktopus_now/logs \
    --surface-extraction ours
