#!/usr/bin/env python
import multiprocessing
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

# sys.path.append('/home/zehaowang/Downloads/on-policy/cooperative_bev_prediction/')
# sys.path.append('/home/zehaowang/Downloads/on-policy/cooperative_bev_prediction/fiery')

# from cooperative_bev_prediction.fiery.multi_ego_trainer import TrainingModule
# from cooperative_bev_prediction.fiery.models.fiery import Fiery
# from cooperative_bev_prediction.fiery.config import get_parser, get_cfg
"""Train script for MPEs."""

from SOGMP_plus.scripts.model import *

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # if all_args.env_name == "MPE":
            #     env = MPEEnv(all_args)
            # else:
            #     print("Can not support the " +
            #           all_args.env_name + "environment.")
            #     raise NotImplementedError
            # env.seed(all_args.seed + rank * 1000)
            # return env
            from onpolicy.envs.env_continuous import ContinuousActionEnv

            scene_id = np.random.randint(1, 4)
            all_args.unity_env_path = f'train_{scene_id}.x86_64'
            all_args.rank = rank
            env = ContinuousActionEnv(all_args)

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # if all_args.env_name == "MPE":
            #     env = MPEEnv(all_args)
            # else:
            #     print("Can not support the " +
            #           all_args.env_name + "environment.")
            #     raise NotImplementedError
            # env.seed(all_args.seed * 50000 + rank * 10000)
            # return env
            from onpolicy.envs.env_continuous import ContinuousActionEnv

            scene_id = np.random.randint(1, 4)
            all_args.unity_env_path = f'train_{scene_id}.x86_64'
            all_args.rank = rank
            all_args.base_port = all_args.base_port + 50
            #all_args.no_graphics = False
            env = ContinuousActionEnv(all_args)
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    # cfg = get_cfg(all_args)
    print(all_args)

    model = RVAEP(input_channels=1, latent_dim=512, output_channels=5)
    model = model.to(device)
    model.eval()

    if all_args.use_prediction:
        checkpoint_path = all_args.prediction_checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        # model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from epoch {}'.format(start_epoch))

    # print(cfg.convert_to_dict())
    # trainer = TrainingModule.load_from_checkpoint(all_args.prediction_checkpoint_path, strict=True)
    # trainer.eval()
    # device = torch.device('cuda:0')
    # trainer.to(device)
    # model = trainer.model
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "prediction_model": model,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.constrained:
            from onpolicy.runner.shared.mpe_runner_cost import MPECRunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])
    
