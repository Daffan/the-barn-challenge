import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo


class Actor(nn.Module):
    def __init__(self, encoder, head, action_dim):
        super(Actor, self).__init__()

        self.encoder = encoder
        self.head = head
        self.fc = nn.Linear(self.encoder.feature_dim, action_dim)

    def forward(self, state):
        a = self.encoder(state) if self.encoder else state
        a = self.head(a)
        return torch.tanh(self.fc(a))


class Critic(nn.Module):
    def __init__(self, encoder, head):
        super(Critic, self).__init__()

        # Q1 architecture
        self.encoder1 = encoder
        self.head1 = head
        self.fc1 = nn.Linear(self.encoder1.feature_dim, 1)

        # Q2 architecture
        self.encoder2 = copy.deepcopy(encoder)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.encoder2.feature_dim, 1)

    def forward(self, state, action):
        state1 = self.encoder1(
            state) if self.encoder1 else state
        sa1 = torch.cat([state1, action], 1)

        state2 = self.encoder2(
            state) if self.encoder2 else state
        sa2 = torch.cat([state2, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.encoder1(
            state) if self.encoder1 else state
        sa = torch.cat([state, action], 1)

        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1


class TD3(BaseRLAlgo):
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
            policy_noise=0.2,
            noise_clip=0.5,
            n_step=4,
            update_actor_freq=2,
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
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
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
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gammas * target_Q

        # Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale  # to range of -1, 1
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self.update_actor_freq == 0:

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