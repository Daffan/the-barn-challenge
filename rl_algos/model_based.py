import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from rl_algos.base_rl_algo import BaseRLAlgo


class Model(nn.Module):
    def __init__(self, encoder, head, state_dim, deterministic=False):
        super(Model, self).__init__()
        self.encoder = encoder
        self.head = head
        self.deterministic = deterministic
        self.history_length, self.state_dim = state_dim
        self.laser_dim = 720
        self.feature_dim = self.state_dim - self.laser_dim
        if not deterministic:
            self.state_dim *= 2  # mean and logvar
            self.laser_dim *= 2
            self.feature_dim *= 2
        
        self.laser_state_fc = nn.Sequential(*[
            nn.Linear(head.feature_dim, self.laser_dim),
            nn.Tanh()
        ])
        self.feature_state_fc = nn.Sequential(*[
            nn.Linear(head.feature_dim, self.feature_dim)
        ])
        
        self.reward_fc = nn.Linear(head.feature_dim, 1)
        self.done_fc = nn.Linear(head.feature_dim, 1)

    def forward(self, state, action):
        s = self.encoder(state) if self.encoder else state
        sa = torch.cat([s, action], 1)
        x = self.head(sa)
        ls = self.laser_state_fc(x)
        fs = self.feature_state_fc(x)
        r = self.reward_fc(x)
        d = F.sigmoid(self.done_fc(x))
        if self.deterministic:
            s = torch.cat([ls, fs], axis=1)
        else:
            s = torch.cat(
                [
                    ls[:, :self.laser_dim // 2], fs[:, :self.feature_dim // 2],
                    ls[:, self.laser_dim // 2:], fs[:, self.feature_dim // 2:]
                ],
                axis=1
            )

        return s, r, d
    
    def sample(self, state, action):
        s, r, d = self.forward(state, action)
        if self.deterministic:
            if self.history_length > 1:
                return torch.cat([state[:, 1:, :], s[:, None, :]], axis=1)
            else:
                return s[:, None, :], r, d
        else:
            mean = s[..., self.state_dim // 2:]
            logvar = s[..., :self.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            if self.history_length > 1:
                return torch.cat([state[:, 1:, :], recon_dist.sample()[:, None, :]], axis=1), r, d
            else:
                return recon_dist.sample()[:, None, :], r, d


class DynaRLAlgo(BaseRLAlgo):
    def __init__(self, model, model_optm, *args, model_update_per_step=5, n_simulated_update=5, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        action -= self._action_bias
        action /= self._action_scale
        done = 1 - not_done
        if self.model.deterministic:
            pred_next_state, r, d = self.model(state, action)
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            pred_next_state_mean_var, r, d = self.model(state, action)
            mean = pred_next_state_mean_var[..., self.model.state_dim // 2:]
            logvar = pred_next_state_mean_var[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()  # nll loss
        
        # pred_reward = self._get_reward(state, pred_next_state[:, None, :])
        # pred_done = self._get_done(pred_next_state[:, None, :])
        reward_loss = self.loss_function(r, reward)
        # done_loss = self.loss_function(d, done.view(-1))
        done_loss = F.binary_cross_entropy(d, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, *_ = replay_buffer.sample(batch_size)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1
        with torch.no_grad():
            for i in range(self.n_step):
                next_action = self.actor_target(state)
                next_action += torch.randn_like(next_action, dtype=torch.float32) * self.exploration_noise  # exploration noise
                if i == 0:
                    action = next_action
                next_state, r, d = self.model.sample(state, next_action)
                reward = r # self._get_reward(state, next_state)[:, None]
                not_done *= (1 - d) # (1 - self._get_done(next_state)[:, None])
                gammas *= self.gamma ** (not_done)
                reward = (reward - replay_buffer.mean) / replay_buffer.std  # reward normalization 
                total_reward = reward + total_reward * gammas

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        rl_loss_info = super().train(replay_buffer, batch_size)
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        for _ in range(self.n_simulated_update):
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas, None)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {}
        loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)
        loss_info.update(simulated_rl_loss_info)

        return loss_info
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class MBPORLAlgo(BaseRLAlgo):
    def __init__(self, model, model_optm, *args, model_update_per_step=5, n_simulated_update=5, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
        self.loss_function = nn.MSELoss()
        self.start_idx = None
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
        action -= self._action_bias
        action /= self._action_scale
        done = 1 - not_done
        if self.model.deterministic:
            pred_next_state, r, d = self.model(state, action)
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            pred_next_state_mean_var, r, d = self.model(state, action)
            mean = pred_next_state_mean_var[..., self.model.state_dim // 2:]
            logvar = pred_next_state_mean_var[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()  # nll loss
        
        # pred_reward = self._get_reward(state, pred_next_state[:, None, :])
        # pred_done = self._get_done(pred_next_state[:, None, :])
        reward_loss = self.loss_function(r, reward)
        # done_loss = self.loss_function(d, done.view(-1))
        done_loss = F.binary_cross_entropy(d, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }

    def simulate_transition(self, replay_buffer, batch_size=256):
        # MBPO only branch from the states visited by the current policy, or newly collected states
        state, _, next_state, reward, *_ = replay_buffer.sample(batch_size, start_idx = self.start_idx)
        total_reward = torch.zeros(reward.shape).to(self.device)
        not_done = torch.ones(reward.shape).to(self.device)
        gammas = 1
        with torch.no_grad():
            for i in range(self.n_step):
                next_action = self.select_action(state, to_cpu=False)
                if i == 0:
                    action = next_action
                next_state, r, d = self.model.sample(state, next_action)
                reward = r # self._get_reward(state, next_state)[:, None]
                not_done *= (1 - d) # (1 - self._get_done(next_state)[:, None])
                gammas *= self.gamma ** (not_done)
                reward = (reward - replay_buffer.mean) / replay_buffer.std  # reward normalization 
                total_reward = reward + total_reward * gammas

        return state, action, next_state, reward, not_done, gammas

    def train(self, replay_buffer, batch_size=256):
        # rl_loss_info = super().train(replay_buffer, batch_size)  # train MBPO purely on model-generated data
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        transitions = []
        # sample all the transition samples before update the policy
        for _ in range(self.n_simulated_update):
            transitions.append(self.simulate_transition(replay_buffer, batch_size))
        for state, action, next_state, reward, not_done, gammas in transitions:
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas)
            simulated_rl_loss_infos.append(simulated_rl_loss_info)
        self.start_idx = replay_buffer.ptr

        simulated_rl_loss_info = {}
        for k in simulated_rl_loss_infos[0].keys():
            simulated_rl_loss_info["simulated" + k] = np.mean([li[k] for li in simulated_rl_loss_infos if li[k] is not None])

        loss_info = {}
        # loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)
        loss_info.update(simulated_rl_loss_info)

        return loss_info
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))


class SMCPRLAlgo(BaseRLAlgo):
    def __init__(self, model, model_optm, *args, horizon=5, num_particle=1024, model_update_per_step=5, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.horizon = horizon
        self.num_particle = num_particle
        self.model_update_per_step = model_update_per_step
        self.loss_function = nn.MSELoss()
        super().__init__(*args, **kw_args)

    def train_model(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done, _, _ = replay_buffer.sample(batch_size)
        action -= self._action_bias
        action /= self._action_scale
        done = 1 - not_done
        if self.model.deterministic:
            pred_next_state, r, d = self.model(state, action)
            state_loss = self.loss_function(pred_next_state, next_state[:, -1, :])
        else:
            pred_next_state_mean_var, r, d = self.model(state, action)
            mean = pred_next_state_mean_var[..., self.model.state_dim // 2:]
            logvar = pred_next_state_mean_var[..., :self.model.state_dim // 2]
            recon_dist = Normal(mean, torch.exp(logvar))
            state_loss = -recon_dist.log_prob(next_state[:, -1, :]).sum(dim=-1).mean()  # nll loss
        
        # pred_reward = self._get_reward(state, pred_next_state[:, None, :])
        # pred_done = self._get_done(pred_next_state[:, None, :])
        reward_loss = self.loss_function(r, reward)
        # done_loss = self.loss_function(d, done.view(-1))
        done_loss = F.binary_cross_entropy(d, done)

        loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "Model_loss": loss.item(),
            "Model_grad_norm": self.grad_norm(self.model)
        }
        
    def train(self, replay_buffer, batch_size=256):
        rl_loss_info = super().train(replay_buffer, batch_size)
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)
        
        loss_info = {}
        loss_info.update(rl_loss_info)
        loss_info.update(model_loss_info)

        return loss_info

    def select_action(self, state):
        if self.exploration_noise >= 0:
            assert len(state.shape) == 2, "does not support batched action selection!"
            state = torch.FloatTensor(state).to(self.device)[None, ...]  # (batch_size=1, history_length, 723)
            s = state.repeat(self.num_particle, 1, 1).clone()
            r = 0
            gamma = torch.zeros((self.num_particle, 1)).to(self.device)
            with torch.no_grad():
                for i in range(self.horizon):
                    # Sample action with policy
                    a = self.actor(s)
                    a += torch.randn_like(a, dtype=torch.float32) * self.exploration_noise
                    if i == 0:
                        a0 = a
                        
                    # simulate trajectories
                    ns, r, d = self.model.sample(s, a)
                    r += r * gamma # self._get_reward(s, ns)[:, None] * gamma
                    gamma *= (1 - d) # (1 - self._get_done(ns)[:, None])
                    s = ns
                q = self.critic.Q1(ns, a)
                r += q * gamma
                
                logit_r = F.softmax(r, -1).view(-1)
                n = Categorical(logit_r).sample()
                a = a0[n]
            return a.cpu().data.numpy()
        else: # deploy the policy only when self.exploration_noise = 0 
            return super().select_action(state)
    
    def save(self, dir, filename):
        super().save(dir, filename)
        self.model.to("cpu")
        with open(join(dir, filename + "_model"), "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        self.model.to(self.device)

    def load(self, dir, filename):
        super().load(dir, filename)
        with open(join(dir, filename + "_model"), "rb") as f:
            self.model.load_state_dict(pickle.load(f))