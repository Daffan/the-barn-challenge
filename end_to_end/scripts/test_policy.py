import os
from os.path import dirname, abspath, join
import yaml
import gym
import torch
import argparse

import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from envs.wrappers import ShapingRewardWrapper, StackFrame
from td3.train import initialize_policy

def get_world_name(config, id):
    assert 0 <= id < 300, "BARN dataset world index ranges from 0-299"
    world_name = "BARN/world_%d.world" %(id)
    return world_name

def load_policy(policy, policy_path):
    policy.load(policy_path, "last_policy")
    policy.exploration_noise = 0
    return policy

def _debug_print_robot_status(env, count, rew, actions):
    Y = env.move_base.robot_config.Y
    X = env.move_base.robot_config.X
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, p.x, X, p.y, Y , rew))

def main(args):
    with open(join(args.policy_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)

    env_config["kwargs"]["world_name"] = world_name
    if args.gui:
        env_config["kwargs"]["gui"] = True
    env_config["kwargs"]["init_sim"] = False
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env)
    policy = load_policy(policy, args.policy_path)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    while ep < args.repeats:
        obs = env.reset()
        ep += 1
        step = 0
        done = False
        while True:
            if not args.default_dwa:
                actions = policy.select_action(obs)
            else:
                actions = env_config["kwargs"]["param_init"]
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            obs = obs_new
            step += 1

            if args.verbose:
                _debug_print_robot_status(env, step, rew, actions)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--world_id', dest='id', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default="end_to_end/data")
    parser.add_argument('--default_dwa', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    main(args)

