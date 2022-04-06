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
from envs.wrappers import StackFrame

BUFFER_PATH = os.getenv('BUFFER_PATH')

# add path to the plugins to the GAZEBO_PLUGIN_PATH
gpp = os.getenv('GAZEBO_PLUGIN_PATH') if os.getenv('GAZEBO_PLUGIN_PATH') is not None else ""
wd = os.getcwd()
os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(wd, "jackal_helper/plugins/build") + ":" + gpp
rospy.logwarn(os.environ['GAZEBO_PLUGIN_PATH'])

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

def load_policy(policy, is_validation):
    f = True
    policy_name = "validation_policy" if is_validation else "policy"
    while f:
        try:
            if not os.path.exists(join(BUFFER_PATH, "%s_copy_actor" %(policy_name))):
                policy.load(BUFFER_PATH, policy_name)
            f = False
        except FileNotFoundError:
            time.sleep(1)
        except:
            logging.exception('')
            time.sleep(1)
    if is_validation:
        policy.exploration_noise = 0
    return policy

def write_buffer(traj, id):
    file_names = os.listdir(join(BUFFER_PATH, 'actor_%s' %(str(id))))
    if len(file_names) == 0:
        ep = 0
    else:
        eps = [int(f.split("_")[-1].split(".pickle")[0]) for f in file_names]  # last index under this folder
        sorted(eps)
        ep = eps[-1] + 1
    print(">>>>>>>>>>>>>>>>>>>>>>>>>", ep)
    if len(file_names) < 10:
        with open(join(BUFFER_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
            try:
                pickle.dump(traj, f)
            except OSError as e:
                logging.exception('Failed to dump the trajectory! %s', e)
                pass
    return ep

def get_world_name(config, id):
    is_validation = id >= config["condor_config"]["num_actor"]
    if not is_validation:
        if len(config["condor_config"]["worlds"]) < config["condor_config"]["num_actor"]:
            duplicate_time = config["condor_config"]["num_actor"] // len(config["condor_config"]["worlds"]) + 1
            worlds = config["condor_config"]["worlds"] * duplicate_time
        else:
            worlds = config["condor_config"]["worlds"].copy()
            random.shuffle(worlds)
            worlds = worlds[:config["condor_config"]["num_actor"]]
        world_name = worlds[id]
    else:
        worlds = config["condor_config"]["validation_worlds"]
        world_id = (id - config["condor_config"]["num_actor"]) % len(worlds)
        assert world_id < len(worlds), "one actor per valiation worlds!"
        world_name = worlds[world_id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name, is_validation

def _debug_print_robot_status(env, count, rew, actions):
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), Y position: %f(world_frame), rew: %f' %(count, p.x, p.y, rew))

def main(id):
    config = initialize_actor(id)
    env_config = config['env_config']
    world_name, is_validation = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env, init_buffer=False)
    bad_traj_count = 0

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    while True:
        obs = env.reset()
        traj = []
        done = False
        policy = load_policy(policy, is_validation)
        while not done:
            actions = policy.select_action(obs)
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([obs, actions, rew, done, info])
            obs = obs_new
            _debug_print_robot_status(env, len(traj), rew, actions)
        
        # time_per_step = info['time'] / len(traj)  # sometimes, the simulation runs very slow, need restart
        # if len(traj) > 1 and time_per_step < (0.05 + config["env_config"]["kwargs"]["time_step"]):
        #     _ = write_buffer(traj, id)
        # else:  # for some reason, the progress might just dead or always give fail traj with only 1 step
        #     bad_traj_count += 1
        #     if bad_traj_count >= 5:
        #         break
        
        write_buffer(traj, id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
