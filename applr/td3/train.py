import argparse
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil
import logging
import collections
import time
from pprint import pformat

import torch

sys.path.append(dirname(dirname(abspath(__file__))))
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame
from td3.information_envs import InfoEnv
from td3.net import MLP, CNN
from td3.rl import Actor, Critic, TD3, ReplayBuffer
from td3.collector import CondorCollector, LocalCollector

def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        dt_string
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer

def initialize_envs(config):
    env_config = config["env_config"]
    
    if not env_config["use_condor"]:
        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        if env_config["shaping_reward"]:
            env = ShapingRewardWrapper(env)
        env = StackFrame(env, stack_frame=env_config["stack_frame"])
    else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
        print("    >>>> Using actors on Condor")
        env = InfoEnv(config)
    return env

def seed(config):
    env_config = config["env_config"]
    
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])

def initialize_policy(config, env):
    training_config = config["training_config"]

    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    # devices = "cpu"  # GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    device = "cpu"  #"cuda:%d" %(devices[0]) if len(devices) > 0 else "cpu"
    print("    >>>> Running on device %s" %(device))

    state_preprocess = CNN(config["env_config"]["stack_frame"]) if training_config["network"] == "cnn" else None
    input_dim = state_preprocess.feature_dim if state_preprocess else np.prod(state_dim) 
    actor = Actor(
        state_preprocess=state_preprocess,
        head=MLP(input_dim, training_config['num_layers'], training_config['hidden_layer_size']),
        action_dim=action_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(), 
        lr=training_config['actor_lr']
    )

    state_preprocess = CNN(config["env_config"]["stack_frame"]) if training_config["network"] == "cnn" else None
    input_dim += np.prod(action_dim)
    critic = Critic(
        state_preprocess,
        head=MLP(input_dim, training_config['num_layers'], training_config['hidden_layer_size'])
    ).to(device)
    critic_optim = torch.optim.Adam(
        critic.parameters(), 
        lr=training_config['critic_lr']
    )

    policy = TD3(
        actor, actor_optim, 
        critic, critic_optim, 
        action_range=[action_space_low, action_space_high],
        device=device,
        **training_config["policy_args"]
    )

    buffer = ReplayBuffer(state_dim, action_dim, training_config['buffer_size'], device=device)

    return policy, buffer

def train(env, policy, buffer, config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    
    if env_config["use_condor"]:
        collector = CondorCollector(policy, env, buffer)
    else:
        collector = LocalCollector(policy, env, buffer)

    training_args = training_config["training_args"]
    print("    >>>> Pre-collect experience")
    collector.collect(n_step=training_config['pre_collect'])

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()
    while n_steps < training_args["max_step"]:
        # Linear decaying exploration noise from "start" -> "end"
        policy.exploration_noise = \
            - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
            *  n_steps / training_args["max_step"] + training_config["exploration_noise_start"]
        steps, epinfo = collector.collect(training_args["collect_per_step"])
        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)
        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        actor_grad_norms = []
        critic_grad_norms = []
        actor_losses = []
        critic_losses = []
        for _ in range(training_args["update_per_step"]):
            actor_grad_norm, critic_grad_norm, actor_loss, critic_loss = policy.train(buffer, training_args["batch_size"])
            if actor_loss is not None:
                actor_grad_norms.append(actor_grad_norm)
                actor_losses.append(actor_loss)
            critic_grad_norms.append(critic_grad_norm)
            critic_losses.append(critic_loss)

        t1 = time.time()
        log = {
            "Episode_return": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Success": np.mean([epinfo["success"] for epinfo in epinfo_buf]),
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "Actor_grad_norm": np.mean(actor_grad_norms),
            "Critic_grad_norm": np.mean(critic_grad_norms),
            "Actor_loss": np.mean(actor_losses),
            "Critic_loss": np.mean(critic_losses),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise
        }
        logging.info(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar('train/' + k, log[k], global_step=n_steps)
            policy.save(save_path, "policy")

            for k in world_ep_buf.keys():
                writer.add_scalar(k + "/Episode_return", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Success", np.mean([epinfo["success"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]), global_step=n_steps)


    if env_config["use_condor"]:
        BASE_PATH = os.getenv('BUFFER_PATH')
        shutil.rmtree(BASE_PATH, ignore_errors=True)  # a way to force all the actors to stop
    else:
        train_envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Start condor training')
    parser.add_argument('--config_path', dest='config_path', default="../configs/config.ymal")
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    seed(config)
    print(">>>>>>>> Creating the environments")
    train_envs = initialize_envs(config)
    env = train_envs if config["env_config"]["use_condor"] else train_envs.env[0]
    
    print(">>>>>>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env)
    print(">>>>>>>> Start training")
    train(train_envs, policy, buffer, config)
