import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import torch
import gym
import numpy as np
import random
import time
import rospy
import argparse
import logging

from td3.train import initialize_policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame

BUFFER_PATH = os.getenv('BUFFER_PATH')

def initialize_actor(id):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' %(str(id)))

    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    while f is None:
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_policy(policy):
    f = True
    while f:
        try:
            if not os.path.exists(join(BUFFER_PATH, "policy_copy_actor")):
                policy.load(BUFFER_PATH, "policy")
            f = False
        except FileNotFoundError:
            time.sleep(1)
        except:
            logging.exception('')
            time.sleep(1)
    return policy

def write_buffer(traj, ep, id):
    with open(join(BUFFER_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        try:
            pickle.dump(traj, f)
        except OSError as e:
            logging.exception('Failed to dump the trajectory! %s', e)
            pass

def get_world_name(config, id):
    if len(config["condor_config"]["worlds"]) < config["condor_config"]["num_actor"]:
        duplicate_time = config["condor_config"]["num_actor"] // len(config["condor_config"]["worlds"]) + 1
        worlds = config["condor_config"]["worlds"] * duplicate_time
    else:
        worlds = config["condor_config"]["worlds"].copy()
        random.shuffle(worlds)
        worlds = worlds[:config["condor_config"]["num_actor"]]
    world_name = worlds[id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name

def _debug_print_robot_status(env, count, rew, actions):
    Y = env.move_base.robot_config.Y
    X = env.move_base.robot_config.X
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, p.x, X, p.y, Y , rew))

def main(id):
    config = initialize_actor(id)
    env_config = config['env_config']
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    while True:
        obs = env.reset()
        ep += 1
        traj = []
        done = False
        policy = load_policy(policy)
        while not done:
            actions = policy.select_action(obs)
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([obs, actions, rew, done, info])
            obs = obs_new

            _debug_print_robot_status(env, len(traj), rew, actions)

        write_buffer(traj, ep, id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
