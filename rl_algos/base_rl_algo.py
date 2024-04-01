from os.path import join
import pickle
import copy
import numpy as np

import torch


class BaseRLAlgo:
    def __init__(
        self, 
        actor,
        actor_optim,
        critic,
        critic_optim,
        action_range,
        n_step=4,
        gamma=0.99,
        device="cpu"
    ):
        self.actor = actor
        self.actor_optimizer = actor_optim

        self.critic = critic
        self.critic_optimizer = critic_optim

        self.n_step = n_step
        self.gamma = gamma

        self.device = device

        self.action_range = action_range
        self._action_scale = torch.tensor(
            (action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor(
            (action_range[1] + action_range[0]) / 2.0, device=self.device)

    def select_action(self, state, to_cpu=True):
        # use the actor network to compute the action
        raise NotImplementedError

    def train_rl(self, state, action, next_state, reward, not_done, gammas):
        # update actor and critic with the batch of transition samples
        raise NotImplementedError

    def train(self, replay_buffer, batch_size=256):
        transitions = replay_buffer.sample_transition(self.n_step, self.gamma, batch_size)
        loss_info = self.train_rl(*transitions)
        return loss_info

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
        self.actor.to(self.device)

    def load(self, dir, filename):
        with open(join(dir, filename + "_actor"), "rb") as f:
            self.actor.load_state_dict(pickle.load(f))
            self.actor_target = copy.deepcopy(self.actor)


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

    def add(self, state, action, next_state, reward, done, task, collision_reward):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # (reward - 0.02478) / 6.499
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task
        self.collision_reward[self.ptr] = collision_reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000 and self.reward_norm:  # and self.mean is None:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2) or self.std is None:
                self.mean, self.std = 0.0, 1.0
        self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size, start_idx=0):
        index = np.random.randint(start_idx, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device),
            torch.FloatTensor(self.task[index]).to(self.device),
            torch.FloatTensor(self.collision_reward[index]).to(self.device),
            index)

    def n_step_return(self, n_step, ind, gamma):
        reward = []
        not_done = []
        next_state = []
        gammas = []
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
                c += self.collision_reward[idx] * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
            collision_reward.append(c)

        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        gammas = torch.FloatTensor(np.array(gammas)).to(self.device)
        
        collision_reward = torch.FloatTensor(
            np.array(collision_reward)
        ).to(self.device)
        return next_state, reward, not_done, gammas, collision_reward

    def sample_transition(self, n_step=4, gamma=0.99, batch_size=256):
        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, not_done, task, collision_reward, index = self.sample(batch_size)
        next_state, reward, not_done, gammas, collision_reward = self.n_step_return(n_step, index, gamma)
        return state, action, next_state, reward, not_done, gammas, collision_reward