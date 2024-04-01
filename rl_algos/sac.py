import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class GaussianActor(nn.Module):
    def __init__(self, encoder, head, action_dim):
        super(GaussianActor, self).__init__()

        self.encoder = encoder
        self.head = head
        self.fc_mean = nn.Linear(self.encoder.feature_dim, action_dim)
        self.fc_log_std = nn.Linear(self.encoder.feature_dim, action_dim)

        self.head.apply(weights_init_)
        self.fc_mean.apply(weights_init_)
        self.fc_log_std.apply(weights_init_)

    def forward(self, state):
        a = self.encoder(state) if self.encoder else state
        a = self.head(a)
        mean = self.fc_mean(a)
        log_std = torch.tanh(self.fc_log_std(a))
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return y_t, log_prob, mean


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


class SAC(BaseRLAlgo):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            device="cpu",
            gamma=0.99,
            n_step=4,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
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
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(np.array(self.action_range).shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0001)

    def select_action(self, state, to_cpu=True):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) < 3:
            state = state[None, :, :]
        action, *_ = self.actor.sample(state)
        if to_cpu:
            action = action.cpu().data.numpy().flatten()
            action *= self._action_scale.cpu().data.numpy()
            action += self._action_bias.cpu().data.numpy()
        return action

    def train_rl(self, state, action, next_state, reward, not_done, gammas, collision_reward):

        with torch.no_grad():
            next_action, next_log_std, _ = self.actor_target.sample(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_std
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

        # Compute actor losse
        action, action_log_std, _ = self.actor.sample(state)
        actor_loss = (self.alpha * action_log_std - self.critic.Q1(state, action)).mean()

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

        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (action_log_std + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        alpha_loss = alpha_loss.item() if self.automatic_entropy_tuning else None
        return {
            "Actor_grad_norm": self.grad_norm(self.actor),
            "Critic_grad_norm": self.grad_norm(self.critic),
            "Actor_loss": actor_loss,
            "Critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
        }

    def save(self, dir, filename):
        super().save(dir, filename)
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(self.alpha, f)

    def load(self, dir, filename):
        super().load(dir, filename)
        self.actor_target = copy.deepcopy(self.actor)
        with open(join(dir, filename + "_noise"), "rb") as f:
            self.alpha = pickle.load(f)