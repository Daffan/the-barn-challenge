########################################################################
# This script tests the jackal navigation environment with random action
########################################################################
import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import random
import numpy as np

import envs.registration
from envs.wrappers import ShapingRewardWrapper

def main():
    
    """
    env = gym.make(
        id='dwa_param_continuous_laser-v0',
        base_local_planner="eband_local_planner/EBandPlannerROS",
        param_init=[0.75],
        param_list=["EBandPlannerROS/max_vel_lin"],
        world_name='BARN/world_114.world',
        gui=True,
        init_position=[-2, 2, np.pi/2],
        goal_position=[0, 10, 0]
    )
    """
    env = gym.make(
        id='dwa_param_continuous_laser-v0', 
        param_init=[0.5, 1.57, 6, 20, 0.75, 1, 0.3],
        param_list=["TrajectoryPlannerROS/max_vel_x", 
                      "TrajectoryPlannerROS/max_vel_theta", 
                      "TrajectoryPlannerROS/vx_samples", 
                      "TrajectoryPlannerROS/vtheta_samples", 
                      "TrajectoryPlannerROS/path_distance_bias", 
                      "TrajectoryPlannerROS/goal_distance_bias", 
                      "inflation_radius"],
        world_name='BARN/world_0.world',
        gui=True,
        init_position=[-2, 2, np.pi/2],
        goal_position=[0, 10, 0]
    )
    """
    env = gym.make(
        id='motion_control_continuous_laser-v0', 
        world_name='world_0.world',
        gui=True,
        init_position=[-2, 2, np.pi/2],
        goal_position=[0, 10, 0],
        time_step=0.2,
        slack_reward=0,
        success_reward=10,
        collision_reward=-10,
        failure_reward=-50
    )
    """
    env = ShapingRewardWrapper(env)
    env.reset()
    done  = False
    count = 0
    ep_rew = 0

    high = env.action_space.high
    low = env.action_space.low
    bias = (high + low) / 2
    scale = (high - low) / 2
    for _ in range(1000): # run 10 steps

        actions = 2*(np.random.rand(env.action_space.shape[0]) - 0.5)
        actions *= scale
        actions += bias

        count += 1
        obs, rew, done, info = env.step(actions)
        ep_rew += rew
        Y = env.move_base.robot_config.Y
        X = env.move_base.robot_config.X
        p = env.gazebo_sim.get_model_state().pose.position
        print('current step: %d, time: %.2f, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, info["time"], p.x, X, p.y, Y , rew))
        print("actions: ", actions)
        # env.visual_costmap(obs)
        if done:
            env.reset()
            print(count, ep_rew)
            count = 0
            ep_rew = 0

    env.close()

if __name__ == '__main__':
    main()
