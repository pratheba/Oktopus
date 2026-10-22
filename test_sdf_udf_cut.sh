#!/bin/bash
export PATH="$PATH:/usr/bin"

module load cuda/12.9
module load gcc/9
source /home/pselvaraju/miniforge3/etc/profile.d/conda.sh
conda activate Oktopus

#python3 test.py   -c config/config_udf_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y udf_adapt_oktopus_puffer_t1.yaml   -s oktopus_9_v1   -o test_udf_grid_256   -r 256  -a udf
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t1.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t1 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t2.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t2 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t3.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t3 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t4.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t4 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t5.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t5 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t6.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t6 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t7.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t7 -r 256  -a sdf &
#python3 test.py   -c config/config_grid_b13d3_oktopus_puffer.yaml   -ckpt final   -y adapt_oktopus_puffer_t8.yaml   -s oktopus_9_v1   -o adapt_oktopus_puffer_t8 -r 256  -a sdf
#python test_nsdudf.py \
#    -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
#    -o adapt_nsdudf_okto_puffer_64 \
#    -s oktopus_9_v1 \
#    -y udf_adapt_oktopus_puffer_body.yaml \
#    -ckpt final \
#    -r 256 \
#    --nsdudf-repo third_party/nsdudf \
#    --nsdudf-grid 65 \
#    --nsdudf-gradient-mode finite_difference \
#    --nsdudf-mesher marching_cubes \
#    --nsdudf-oracle-chunk-size 8192 \
#    --nsdudf-offset-world 0.1 \
#    --udf-batch-size 32768 \
#    --udf-cleanup

python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o test_adapt_oktopus_puffer_sdf_udf_cut_combined_mc512 \
  -s oktopus_9_v1 \
  -y adapt_oktopus_puffer_blend_udf_cut.yaml \
  -r 512 \
  -ckpt final \
  --mc-mesh "inference/4M/adapt_oktopus_puffer_optim_512/oktopus_9_v1/{index}_{mode}_{accessory}.ply" \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768 \
  --udf-cut-seed-world 0.02 \
  --udf-cut-grow-world 0.01 \
  --udf-cut-min-faces 8 \
  --udf-cut-min-seed-faces 2 \
  --udf-cut-min-seed-fraction 0.05 \
  --udf-cut-cleanup-boundary \
  --udf-cut-boundary-smooth-iterations 4 \
  --udf-cut-boundary-smooth-lambda 0.45 \
  --udf-cut-boundary-smooth-mu -0.47 \
  --udf-cut-boundary-max-step-fraction 0.20 \
  --udf-cut-boundary-max-total-fraction 0.65


python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0067_direct_udf_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0067_final.yaml \
  -r 512 \
  -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768


#python test_sdf_udf_cut.py \
#  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
#  -o test_adapt_oktopus_puffer_sdf_udf_cut_smooth \
#  -s oktopus_9_v1 \
#  -y adapt_oktopus_puffer.yaml \
#  -r 128 \
#  -ckpt final \
#  --mc-mesh "inference/4M/adapt_oktopus_puffer/oktopus_9_v1/oktopus_9_v1_adapt.ply" \
#  --udf-far-value 0.1 \
#  --udf-model-batch-size 32768 \
#  --udf-cut-query-batch-size 32768 \
#  --udf-cut-seed-world 0.02 \
#  --udf-cut-grow-world 0.01 \
#  --udf-cut-min-faces 8 \
#  --udf-cut-min-seed-faces 2 \
#  --udf-cut-min-seed-fraction 0.05 \
#  --udf-cut-cleanup-boundary \
#  --udf-cut-boundary-smooth-iterations 12 \
#  --udf-cut-boundary-smooth-lambda 0.45 \
#  --udf-cut-boundary-smooth-mu -0.47 \
#  --udf-cut-boundary-max-step-fraction 0.40 \
#  --udf-cut-boundary-max-total-fraction 0.65
