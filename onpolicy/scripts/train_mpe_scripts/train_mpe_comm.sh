#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=3
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1
episode_length=200
use_centralized_V=True

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python scripts/train/train_mpe.py --num_agents ${num_agents} --episode_length ${episode_length} --use_centralized_V ${use_centralized_V} --wandb_name "marl" --user_name "liyiping-University of California, Riverside"
done