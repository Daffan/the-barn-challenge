import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.td3 import TD3


class SafeTD3(TD3):
    def __init__(
        self,
        safe_critic,
        safe_critic_optim,
        safe_threshold=-0.1,
        safe_lagr=0.1,
        safe_mode="lagr",
        *args, **kw_args
    ):
        super().__init__(*args, **kw_args)
        self.safe_lagr = safe_lagr
        self.safe_mode = safe_mode
        self.safe_critic = safe_critic
        self.safe_critic_target = copy.deepcopy(self.safe_critic)
        self.safe_critic_optimizer = safe_critic_optim
        self.safe_threshold = safe_threshold

        self.grad_dims = [p.numel() for p in self.actor.parameters()]
        n_params = sum(self.grad_dims)
        self.grads = torch.zeros((n_params, 2)).to(self.device)

    def grad2vec(self, grad, i):
        self.grads[:,i].fill_(0.0)
        beg = 0
        for p, g, dim in zip(self.actor.parameters(), grad, self.grad_dims):
            en = beg + dim
            if g is not None:
                self.grads[beg:en,i].copy_(g.view(-1).data.clone())
            beg = en

    def vec2grad(self, grad):
        beg = 0
        for p, dim in zip(self.actor.parameters(), self.grad_dims):
            en = beg + dim
            p.grad = grad[beg:en].data.clone().view(*p.shape)
            beg = en

    def safe_update(self, neg_safe_advantage):
        g1 = self.grads[:,0]
        g2 = -self.grads[:,1]
        phi = neg_safe_advantage.detach() - self.safe_threshold
        lmbd = F.relu((0.1 * phi - g1.dot(g2))/(g2.dot(g2)+1e-8))
        return g1 + lmbd * g2

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

            # Compute the target safe Q value
            safe_target_Q1, safe_target_Q2 = self.safe_critic_target(next_state, next_action)
            safe_target_Q = torch.min(safe_target_Q1, safe_target_Q2)
            safe_target_Q = collision_reward + not_done * gammas * safe_target_Q

        # Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale  # to range of -1, 1
        current_Q1, current_Q2 = self.critic(state, action)
        safe_current_Q1, safe_current_Q2 = self.safe_critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        safe_critic_loss = F.mse_loss(safe_current_Q1, safe_target_Q) + \
            F.mse_loss(safe_current_Q2, safe_target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.safe_critic_optimizer.zero_grad()
        safe_critic_loss.backward()
        self.safe_critic_optimizer.step()

        actor_loss = None
        safe_actor_loss = None

        # Delayed policy updates
        if self.total_it % self.update_actor_freq == 0:

            # Compute actor losse
            action_now = self.actor(state)
            actor_loss = -self.critic.Q1(state, action_now).mean()
            safe_actor_loss = -self.safe_critic.Q1(state, action_now).mean()

            # Optimize the actor

            if self.safe_mode == "lagr": # use the lagrangian method
                self.actor_optimizer.zero_grad()
                actor_loss = actor_loss + self.safe_lagr * safe_actor_loss
                actor_loss.backward()
                self.actor_optimizer.step()
            elif self.safe_mode == "lyapunov": # use the lyapunov method
                self.actor_optimizer.zero_grad()
                grad_1 = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)
                self.grad2vec(grad_1, 0)
                grad_2 = torch.autograd.grad(safe_actor_loss, self.actor.parameters())
                self.grad2vec(grad_2, 1)
                grad = self.safe_update(safe_actor_loss)
                self.vec2grad(grad)
                self.actor_optimizer.step()
            else:
                raise Exception(f"[error] Unknown safe mode {self.safe_mode}!")

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.safe_critic.parameters(), self.safe_critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        actor_loss = actor_loss.item() if actor_loss is not None else None
        safe_actor_loss = safe_actor_loss.item() if safe_actor_loss is not None else None
        critic_loss = critic_loss.item()
        return {
            "Actor_grad_norm": self.grad_norm(self.actor),
            "Critic_grad_norm": self.grad_norm(self.critic),
            "Safe_critic_norm": self.grad_norm(self.safe_critic),
            "Actor_loss": actor_loss,
            "Safe_actor_loss": safe_actor_loss,
            "Critic_loss": critic_loss,
            "safe_critic_loss": safe_critic_loss.item()
        }