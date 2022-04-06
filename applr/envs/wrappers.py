import gym
import numpy as np
from collections import deque

class ShapingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        self.Y = self.env.gazebo_sim.get_model_state().pose.position.y
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += position.y - self.Y
        self.Y = position.y
        return obs, rew, done, info

class StackFrame(gym.Wrapper):
    def __init__(self, env, stack_frame=1):
        super().__init__(env)
        self.stack_frame = stack_frame

    def reset(self):
        self.frames = deque(maxlen=self.stack_frame)
        obs = self.env.reset()
        self.frames.extend([obs]*self.stack_frame)
        return np.stack(self.frames)

    def step(self, *args, **kwargs):
        obs, rew, done, info = self.env.step(*args, **kwargs)
        self.frames.append(obs)
        return np.stack(self.frames), rew, done, info
