#!/bin/sh
env="unity"
scenario="3_scenes"
num_landmarks=3
num_agents=3
algo="rmappo" #"mappo" "ippo"
exp="3_scenes_collect_data"

unity_env_path="./train_1.x86_64"
base_port=18006
seed_max=1
no_graphics=False

episode_length=1000
num_env_steps=150000
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

# use_prediction=True
coop_mode=no_coop
save_data_dir='../SOGMP_plus/datasets_3_scenes'

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
    --eval_episodes ${eval_episodes} \
    --num_mini_batch ${num_mini_batch} \
    --use_eval ${use_eval} \
    --eval_interval ${eval_interval} \
    --save_interval ${save_interval} \
    --use_centralized_V ${use_centralized_V} \
    --num_env_steps ${num_env_steps} \
    --coop_mode ${coop_mode} \
    --collect_data \
    --save_data_dir ${save_data_dir} \
    --ego_vec_size 6 \
    --other_vec_size 12 \
    --human_num 10 \
    --human_vec_size 90 \
    --resolution 0.1 \
    # --model_dir ${model_dir} \
    # --visualization \
    # --use_gt_ogm True \
    # --use_prediction ${use_prediction} \
    # --prediction_checkpoint_path ${prediction_checkpoint_path} \
    # --uncertainty_type ${uncertainty_type} \
    # --use_feature_normalization ${use_feature_normalization} \
    # --use_valuenorm ${use_valuenorm} \
    # --entropy_coef ${entropy_coef} \
    # --clip_param ${clip_param} \
done