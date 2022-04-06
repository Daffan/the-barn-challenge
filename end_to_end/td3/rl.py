import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class Actor(nn.Module):
    def __init__(self, state_preprocess, head, action_dim):
        super(Actor, self).__init__()

        self.state_preprocess = state_preprocess
        self.head = head
        self.fc = nn.Linear(self.state_preprocess.feature_dim, action_dim)

    def forward(self, state):
        a = self.state_preprocess(state) if self.state_preprocess else state
        a = self.head(a)
        return torch.tanh(self.fc(a))


class Critic(nn.Module):
    def __init__(self, state_preprocess, head):
        super(Critic, self).__init__()

        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        self.fc1 = nn.Linear(self.state_preprocess1.feature_dim, 1)

        # Q2 architecture
        self.state_preprocess2 = copy.deepcopy(state_preprocess)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.state_preprocess2.feature_dim, 1)

    def forward(self, state, action):
        state1 = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa1 = torch.cat([state1, action], 1)

        state2 = self.state_preprocess2(
            state) if self.state_preprocess2 else state
        sa2 = torch.cat([state2, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa = torch.cat([state, action], 1)

        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1


class Model(nn.Module):
    def __init__(self, state_preprocess, head, state_dim, deterministic=False):
        super(Model, self).__init__()
        self.state_preprocess = state_preprocess
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
        s = self.state_preprocess(state) if self.state_preprocess else state
        sa = torch.cat([s, action], 1)
        x = self.head(sa)
        ls = self.laser_state_fc(x)
        fs = self.feature_state_fc(x)
        r = self.reward_fc(x)
        d = F.sigmoid(self.done_fc(x))
        if self.deterministic:
            s = torch.cat([ls, fs], axis=1)
        else:
            s = torch.cat([ls[:, :self.laser_dim // 2], fs[:, :self.feature_dim // 2], ls[:, self.laser_dim // 2:], fs[:, self.feature_dim // 2:]], axis=1)

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

class TD3(object):
    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            safe_critic=None,
            safe_critic_optim=None,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            n_step=4,
            update_actor_freq=2,
            exploration_noise=0.1,
            safe_threshold=-0.1,
            safe_lagr=0.1,
            safe_mode="lagr",
    ):

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = actor_optim

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = critic_optim

        if safe_critic is not None:
            self.safe_rl = True
            self.safe_lagr = safe_lagr
            self.safe_mode = safe_mode
            self.safe_critic = safe_critic
            self.safe_critic_target = copy.deepcopy(self.safe_critic)
            self.safe_critic_optimizer = safe_critic_optim
            self.safe_threshold = safe_threshold
        else:
            self.safe_rl = False

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.exploration_noise = exploration_noise
        self.device = device
        self.n_step = n_step

        self.total_it = 0
        self.action_range = action_range
        self._action_scale = torch.tensor(
            (action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor(
            (action_range[1] + action_range[0]) / 2.0, device=self.device)

        if self.safe_rl:
            self.grad_dims = [p.numel() for p in self.actor.parameters()]
            n_params = sum(self.grad_dims)
            self.grads = torch.zeros((n_params, 2)).to(self.device)

    def grad2vec(self, grad, i):
        assert self.safe_rl, "[error] Not in Safe RL setting!"
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

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) < 3:
            state = state[None, :, :]
        action = self.actor(state).cpu().data.numpy().flatten()
        action += np.random.normal(0, self.exploration_noise, size=action.shape)
        action *= self._action_scale.cpu().data.numpy()
        action += self._action_bias.cpu().data.numpy()
        return action

    def sample_transition(self, replay_buffer, batch_size=256):
        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, ind = replay_buffer.sample(
            batch_size)
        next_state, reward, not_done, gammas = replay_buffer.n_step_return(self.n_step, ind, self.gamma)
        return state, action, next_state, reward, not_done, gammas

    def train_rl(self, state, action, next_state, reward, not_done, gammas):
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

    def train(self, replay_buffer, batch_size=256):
        if self.safe_rl:
            return self.safe_train(replay_buffer, batch_size)
        state, action, next_state, reward, not_done, gammas = self.sample_transition(replay_buffer, batch_size)
        loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas)
        return loss_info

    def safe_train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, collision_reward, ind = replay_buffer.sample(
            batch_size)

        next_state, reward, not_done, gammas, collision_reward = replay_buffer.n_step_return(self.n_step, ind, self.gamma)

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
            else: # use the lyapunov method
                self.actor_optimizer.zero_grad()
                grad_1 = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)
                self.grad2vec(grad_1, 0)
                grad_2 = torch.autograd.grad(safe_actor_loss, self.actor.parameters())
                self.grad2vec(grad_2, 1)
                grad = self.safe_update(safe_actor_loss)
                self.vec2grad(grad)
                self.actor_optimizer.step()

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
        critic_loss = critic_loss.item()
        return {
            "Actor_grad_norm": self.grad_norm(self.actor),
            "Critic_grad_norm": self.grad_norm(self.critic),
            "Safe_critic_norm": self.grad_norm(self.safe_critic),
            "Actor_loss": actor_loss,
            "Safe_actor_loss": safe_actor_loss.item() if safe_actor_loss else None,
            "Critic_loss": critic_loss,
            "safe_critic_loss": safe_critic_loss.item()
        }

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def save(self, dir, filename):
        self.actor.to("cpu")
        with open(join(dir, filename + "_actor"), "wb") as f:
            pickle.dump(self.actor.state_dict(), f)
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(self.exploration_noise, f)
        self.actor.to(self.device)

    def load(self, dir, filename):
        with open(join(dir, filename + "_actor"), "rb") as f:
            self.actor.load_state_dict(pickle.load(f))
            self.actor_target = copy.deepcopy(self.actor)
        with open(join(dir, filename + "_noise"), "rb") as f:
            self.exploration_noise = pickle.load(f)


class DynaTD3(TD3):
    def __init__(self, model, model_optm, model_update_per_step, n_simulated_update, *args, **kw_args):
        self.model = model
        self.model_optimizer = model_optm
        self.model_update_per_step = model_update_per_step
        self.n_simulated_update = n_simulated_update
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
    
    def _get_reward(self, state, next_state):
        # This is a hard-coded reward function!!
        rew = next_state[:, -1, -5] / 2  # this is delta y
        d, _ = torch.sort(next_state[:, -1, :720] * 2 + 2, axis=-1)
        d = d[:, :10].mean(axis=-1, keepdim=False)
        rew += -torch.ones_like(d) * (d < 0.3)  # hard-coded collision coefficient!!
        rew += (next_state[:, -1, -4] > 1).long() * 20
        return rew

    def _get_done(self, next_state):
        d, _ = torch.sort(next_state[:, -1, :720] * 2 + 2, axis=-1)
        d = d[:, :10].mean(axis=-1, keepdim=False)
        return torch.logical_or(next_state[:, -1, -4] > 1.4, d < 0.3).long()

    def train(self, replay_buffer, batch_size=256):
        rl_loss_info = super().train(replay_buffer, batch_size)
        for _ in range(self.model_update_per_step):
            model_loss_info = self.train_model(replay_buffer, batch_size)

        simulated_rl_loss_infos = []
        for _ in range(self.n_simulated_update):
            state, action, next_state, reward, not_done, gammas = self.simulate_transition(replay_buffer, batch_size)
            simulated_rl_loss_info = self.train_rl(state, action, next_state, reward, not_done, gammas)
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


class SMCPTD3(TD3):
    def __init__(self, model, model_optm, horizon, num_particle, model_update_per_step, *args, **kw_args):
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
    
    def _get_reward(self, state, next_state):
        # This is a hard-coded reward function!!
        rew = next_state[:, -1, -5] / 2  # this is delta y
        d, _ = torch.sort(next_state[:, -1, :720] * 2 + 2, axis=-1)
        d = d[:, :10].mean(axis=-1, keepdim=False)
        rew += -torch.ones_like(d) * (d < 0.1)  # hard-coded collision coefficient!!
        rew += (next_state[:, -1, -4] > 1).long() * 20
        return rew

    def _get_done(self, next_state):
        d, _ = torch.sort(next_state[:, -1, :720] * 2 + 2, axis=-1)
        d = d[:, :10].mean(axis=-1, keepdim=False)
        return torch.logical_or(next_state[:, -1, -4] > 1.4, d < 0.1).long()
    
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


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu", safe_rl=False, reward_norm=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0
        self.reward_norm = reward_norm

        self.safe_rl = safe_rl

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.collision_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))

        # Reward normalization
        self.mean = None
        self.std = None

        self.device = device

    def add(self, state, action, next_state, reward, done, task, collision_reward=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # (reward - 0.02478) / 6.499
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task

        if self.safe_rl:
            assert collision_reward is not None, "collision reward should not be None"
            self.collision_reward[self.ptr] = collision_reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000 and self.reward_norm:  # and self.mean is None:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2) or self.std is None:
                self.mean, self.std = 0.0, 1.0
        self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.safe_rl:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                torch.FloatTensor(self.collision_reward[ind]).to(self.device),
                ind)
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                ind)

    def n_step_return(self, n_step, ind, gamma):
        reward = []
        not_done = []
        next_state = []
        gammas = []
        if self.safe_rl:
            collision_reward = []

        for i in ind:
            n = 0
            r = 0
            c = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                assert self.mean is not None
                assert self.std is not None
                r += (self.reward[idx] - self.mean) / self.std * gamma ** n
                if self.safe_rl:
                    c += self.collision_reward[idx] * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
            if self.safe_rl:
                collision_reward.append(c)

        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        gammas = torch.FloatTensor(np.array(gammas)).to(self.device)
        if self.safe_rl:
            collision_reward = torch.FloatTensor(
                    np.array(collision_reward)).to(self.device)
        if self.safe_rl:
            return next_state, reward, not_done, gammas, collision_reward
        else:
            return next_state, reward, not_done, gammas
