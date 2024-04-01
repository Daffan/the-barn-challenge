import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo


class DDPG(BaseRLAlgo):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            n_step=4,
            exploration_noise=0.1,
    ):
        super().__init__(
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            n_step,
            gamma,
            device
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.tau = tau
        self.exploration_noise = exploration_noise

        self.total_it = 0

    def select_action(self, state, to_cpu=True):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) < 3:
            state = state[None, :, :]
        action = self.actor(state)
        action += torch.randn_like(action) * self.exploration_noise
        if to_cpu:
            action = self.actor(state).cpu().data.numpy().flatten()
            action *= self._action_scale.cpu().data.numpy()
            action += self._action_bias.cpu().data.numpy()
        return action

    def train_rl(self, state, action, next_state, reward, not_done, gammas, collision_reward):
        self.total_it += 1

        with torch.no_grad():
            next_action = self.actor_target(next_state).clamp(-1, 1)

            # Compute the target Q value
            target_Q = self.critic_target.Q1(next_state, next_action)
            target_Q = reward + not_done * gammas * target_Q

        # Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale  # to range of -1, 1
        current_Q = self.critic.Q1(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Compute actor losse
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        return {
            "Actor_grad_norm": self.grad_norm(self.actor),
            "Critic_grad_norm": self.grad_norm(self.critic),
            "Actor_loss": actor_loss,
            "Critic_loss": critic_loss
        }

    def save(self, dir, filename):
        super().save(dir, filename)
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(self.exploration_noise, f)

    def load(self, dir, filename):
        super().load(dir, filename)
        self.actor_target = copy.deepcopy(self.actor)
        with open(join(dir, filename + "_noise"), "rb") as f:
            self.exploration_noise = pickle.load(f)