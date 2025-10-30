from utils import Actor, Double_Q_Critic, Action_adapter
import torch.nn.functional as F
import numpy as np
import torch
import copy
import random
from state_predict import StateNetwork
from reward_predict import RewardNetwork



class EmptyListError(Exception):
    pass

class SAC_countinuous():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self,number):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        device = s.device  
        state_dim = s.size(1)  
        action_dim = a.size(1)  #
        state_model = StateNetwork((state_dim, state_dim),(action_dim, state_dim),0.5,state_dim,1e-11,0.01,hard=True)
        state_model.load_parameters("./next_state/state_{}.pth".format(number), device)
        state_model = state_model.to(device).eval()
        reward_model = RewardNetwork((state_dim, 1), (action_dim, 1), 0.6,1, 1e-11,0.01, hard=False)
        reward_model.load_parameters("./reward/reward_{}.pth".format(number), device)
        reward_model = reward_model.to(device).eval()

        s, a, r, s_next = self.modify_state(s, a, r, s_next, 20, state_model, reward_model)
        

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def test(self, env, num_episodes=10):
        
        total_rewards = []
        for _ in range(num_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(state, deterministic=True)  # Use deterministic actions for testing
                action = Action_adapter(action, 5)  
                state, reward, done, _, info  = env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        return avg_reward

    def save(self,EnvName, timestep,number):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}_{}.pth".format(EnvName,timestep,number))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}_{}.pth".format(EnvName,timestep,number))

    def load(self,EnvName, timestep,number):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}_{}.pth".format(EnvName, timestep,number)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}_{}.pth".format(EnvName, timestep,number)))
        
    def modify_state(self, states, actions, rewards, next_states, number, state_model, reward_model):
        device = states.device 
        state_dim = states.size(1) 

        indices = random.sample(range(len(states)), number)  
        selected_dimensions = torch.randint(0, state_dim, (number,))  
        # selected_dimensions = torch.tensor([2]*number)

        selected_states = states[indices]  # [number, state_dim]
        selected_actions = actions[indices]  # [number, action_dim]

        expanded_selected_states = selected_states.unsqueeze(1)  # [number, 1, state_dim]
        states_expanded = states.unsqueeze(0)  # [1, num_states, state_dim]

        similarities = []
        for i, dim in enumerate(selected_dimensions):
            mask = torch.ones(state_dim, dtype=torch.bool, device=device)  
            mask[dim] = False  

            selected_state_filtered = expanded_selected_states[i, :, mask]  # [1, state_dim-1]
            states_filtered = states_expanded[0, :, mask]  # [num_states, state_dim-1]

            similarity = torch.norm(states_filtered - selected_state_filtered, dim=1)  # [num_states]
            similarities.append(similarity.unsqueeze(0))  

        similarities = torch.cat(similarities, dim=0)  # [number, num_states]

        differences = []
        for i, dim in enumerate(selected_dimensions):
            diff = torch.abs(states_expanded[0, :, dim] - expanded_selected_states[i, 0, dim])  # [num_states]
            differences.append(diff.unsqueeze(0)) 
        differences = torch.cat(differences, dim=0)  # [number, num_states]

        data = differences / (similarities + 1e-8)  

        k_indices = torch.argmax(data, dim=1)  # [number]
        modified_states = selected_states.clone()
        for i in range(number):
            modified_states[i, selected_dimensions[i]] = states[k_indices[i], selected_dimensions[i]]
        batch_states = modified_states.to(device)
        batch_actions = selected_actions.to(device)

        with torch.no_grad():
            batch_next_states = state_model(batch_states, batch_actions)[0]
            batch_rewards = reward_model(batch_states, batch_actions)[0]

        next_states[indices] = batch_next_states
        rewards[indices] = batch_rewards

        return states, actions, rewards, next_states



class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
   
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
        self.r[self.ptr] = torch.tensor(r, device=self.dvc)
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
    
    def modify_state(self, states, actions, rewards, next_states, number):
        index = random.sample(range(len(states)), number)
        out_states = []
        out_actions = []
        out_next_states = []
        out_rewards = []
        for s_t_index in index:
            selected_dimension = random.randint(0, len(states[0])-1)
            s_t = states[s_t_index]
            similarities = torch.norm(s_t.unsqueeze(0).expand_as(states)[:, :selected_dimension] - 
                                      states[:, :selected_dimension], dim=1, p=2) + \
                           torch.norm(s_t.unsqueeze(0).expand_as(states)[:, selected_dimension+1:] - 
                                      states[:, selected_dimension+1:], dim=1, p=2)
            differences = torch.abs(s_t[selected_dimension] - states[:, selected_dimension])
            data = differences / similarities
            sorted_data, sorted_indices = torch.sort(data, descending=True)
            nan_indices = torch.isnan(sorted_data)
            first_non_nan_index = (nan_indices == False).nonzero(as_tuple=True)[0][0]
            original_indices = (data == sorted_data[first_non_nan_index]).nonzero(as_tuple=True)[0]
            k = original_indices[0].item()
            s_t_prime = s_t.clone()
            s_t_prime[selected_dimension] = states[k, selected_dimension]
            out_states.append(s_t_prime)
            out_next_states.append(next_states[s_t_index])
            out_actions.append(actions[s_t_index])
            out_rewards.append(rewards[s_t_index])
        return torch.stack(out_states), torch.stack(out_actions), torch.stack(out_rewards), torch.stack(out_next_states)
    
    def save_buffer(self, filepath):
        data = {
            'states': self.s[:self.size],
            'actions': self.a[:self.size],
            'rewards': self.r[:self.size],
            'next_states': self.s_next[:self.size],
            'dones': self.dw[:self.size]
        }
        torch.save(data, filepath)
        
    def reward_sample(self, reward_size): 
        dones = self.dw.squeeze()
        trajectories = []
        done_indices = (dones == True).nonzero(as_tuple=True)[0]

        if len(done_indices) == 0:
            return trajectories  
        if done_indices[0].item() > 0:
            start_idx = 0
        else:
            start_idx = 1  
        for i in range(len(done_indices)):
            end_idx = done_indices[i].item()
            if start_idx <= end_idx:
                trajectory = {
                    'states': self.s[start_idx:end_idx + 1],
                    'actions': self.a[start_idx:end_idx + 1],
                    'rewards': self.r[start_idx:end_idx + 1],
                }
                trajectories.append(trajectory)
            start_idx = end_idx + 1  

        if len(trajectories) < reward_size:
            return trajectories
        return random.sample(trajectories, reward_size)        