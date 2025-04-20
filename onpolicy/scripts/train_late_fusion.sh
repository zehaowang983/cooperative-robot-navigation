#!/bin/sh
env="unity"
scenario="3_scenes"
num_landmarks=3
num_agents=3
algo="rmappo" #"mappo" "ippo"
exp="late_fusion"

unity_env_path="./train_1.x86_64"
base_port=10006
seed_max=1
no_graphics=False

# episode_length=150
episode_length=1000
num_env_steps=1500000
# data_chunk_length=15
data_chunk_length=20
num_mini_batch=10
eval_episodes=1
use_eval=True
eval_interval=10
save_interval=10

use_centralized_V=False
use_feature_normalization=False
use_valuenorm=False

entropy_coef=0.0
clip_param=0.05
# lr=4e-5
# critic_lr=4e-5

# constrained=True
# use_prediction=True
# use_traj=True
coop_mode=late
# prediction_checkpoint_path='../SOGMP_plus/models_0.2s_no_coop_5/model15.pth'

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python scripts/train/train_unity.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --n_training_threads 1 \
    --n_rollout_threads 10 \
    --n_eval_rollout_threads 1 \
    --seed ${seed} \
    --unity_env_path ${unity_env_path} \
    --base_port ${base_port} \
    --data_chunk_length ${data_chunk_length} \
    --episode_length ${episode_length} \
    --num_mini_batch ${num_mini_batch} \
    --eval_episodes ${eval_episodes} \
    --use_eval ${use_eval} \
    --eval_interval ${eval_interval} \
    --save_interval ${save_interval} \
    --use_centralized_V ${use_centralized_V} \
    --num_env_steps ${num_env_steps} \
    --coop_mode ${coop_mode} \
    --ego_vec_size 6 \
    --other_vec_size 12 \
    --human_num 10 \
    --human_vec_size 90 \
    --resolution 0.1 \
    # --lr ${lr} \
    # --critic_lr ${critic_lr}
    # --use_prediction ${use_prediction} \
    # --prediction_checkpoint_path ${prediction_checkpoint_path} \
    # --use_wandb \
    # --constrained ${constrained} \
    # --use_feature_normalization ${use_feature_normalization} \
    # --use_valuenorm ${use_valuenorm} \
    # --entropy_coef ${entropy_coef} \
    # --clip_param ${clip_param} \
done