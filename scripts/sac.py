import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from tqdm.notebook import tqdm

import pandas as pd

from scripts.save_everything import save_frames_as_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
figure_save_dir = './figures'
if not os.path.exists(figure_save_dir):
    os.makedirs(figure_save_dir)
    
# I have to say that normally SAC is used for """continuous""" action spaces, but it can also be adapted for discrete action spaces.
# For discrete action spaces, we can use a softmax policy

# Define the neural network architecture for SAC
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ContinuousActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log std output heads
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create a normal distribution with the mean and std
        normal = Normal(mu, std)
        
        # Sample from the distribution
        x_t = normal.rsample()  # rsample: reparameterization trick
        
        # Squash the action using tanh to bound it to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Calculate the log probability of the action
        action = y_t
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state):
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
            action, _ = self.sample(state)
            return action.cpu().numpy()[0]
        
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DiscreteActor, self).__init__()
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def sample(self, state):
        logits = self.forward(state)
        # Use softmax to get probabilities
        prob = F.softmax(logits, dim=-1)
        # Sample from the categorical distribution
        action = torch.multinomial(prob, num_samples=1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)
        action_one_hot = F.one_hot(action.squeeze(1), num_classes=self.action_dim).float()
        return action_one_hot, log_prob
    
    def get_action(self, state):
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1).cpu().numpy()[0]
            return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, is_discrete=False):
        super(Critic, self).__init__()
        self.is_discrete = is_discrete
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        if self.is_discrete and action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=action.shape[-1]).float()
        
        # Concatenate state and action
        sa = torch.cat([state, action], 1)
        
        # Q1 Value
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2 Value
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done):
        # Store transition in buffer
        self.state[self.idx] = state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_state[self.idx] = next_state
        self.done[self.idx] = done
        
        # Update pointer
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        # Sample batch_size elements from buffer
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[indices]).to(device),
            torch.FloatTensor(self.action[indices]).to(device),
            torch.FloatTensor(self.reward[indices]).to(device),
            torch.FloatTensor(self.next_state[indices]).to(device),
            torch.FloatTensor(self.done[indices]).to(device)
        )
    
    def __len__(self):
        return self.size

# SAC Agent
class SAC:
    def __init__(
        self, 
        state_dim,
        action_dim,
        action_space,
        lr=3e-4,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        auto_entropy_tuning=True,
        is_discrete=False,
        device=device
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_space = action_space
        self.is_discrete = is_discrete
        
        if is_discrete:
            self.actor = DiscreteActor(state_dim, action_dim, hidden_dim).to(device)
        else:
            self.actor = ContinuousActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim, is_discrete).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, is_discrete).to(device)
        
        # Initialize target network with same weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            if is_discrete:
                self.target_entropy = -np.log(1.0 / action_space.n)
            else:
                # For continuous actions, target entropy is -dim of action space
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
    def select_action(self, state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0)
                if isinstance(self.actor, DiscreteActor):
                    logits = self.actor(state)
                    action = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    return action
                elif isinstance(self.actor, ContinuousActor):
                # For continuous actions, sample from the actor
                    mu, _ = self.actor(state)
                    return torch.tanh(mu).cpu().numpy()[0]
        else:
            return self.actor.get_action(state)
    
    def update_parameters(self):
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        with torch.no_grad():
            # Get next actions and their log probabilities
            next_action, next_log_prob = self.actor.sample(next_state)
            
            # Get Q values for next states from target critics
            next_q1_target, next_q2_target = self.critic_target(next_state, next_action)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Compute target Q value (Bellman equation)
            next_q_value = reward + (1 - done) * self.gamma * (next_q_target - self.alpha * next_log_prob)
        
        # Get current Q estimates from critics
        q1, q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter alpha
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, f"{directory}/{filename}")
        
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# Training loop
def train(env_name, agent, num_episodes, max_steps, eval_interval=10):
    env = gym.make(env_name)
    env.reset(seed=42)
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in tqdm(range(1, num_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1, max_steps + 1):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store the transition in the replay buffer
            agent.buffer.push(state, action, reward, next_state, float(done))
            
            # Update the parameters
            agent.update_parameters()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        # Evaluate the agent
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_rewards.append(avg_reward)
            print(f"Episode: {episode}, Avg. Reward: {avg_reward:.2f}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_episodes, eval_interval), avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'SAC Learning Curve - {env_name}')
    plt.grid(True)
    plt.savefig(f"{figure_save_dir}/sac_{env_name}_learning_curve.png")
    plt.show()
    
    return episode_rewards, avg_rewards


def train_record(env_name, agent, num_episodes, max_steps, eval_interval=10, record_interval=10):
    env = gym.make(env_name)
    env.reset(seed=42)
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in tqdm(range(1, num_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1, max_steps + 1):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if agent.is_discrete:
                action = np.zeros(agent.action_space.n, dtype=np.float32)
                action[np.argmax(action)] = 1.0  # Convert to one-hot encoding
            # Store the transition in the replay buffer
            agent.buffer.push(state, action, reward, next_state, float(done))
            
            # Update the parameters
            agent.update_parameters()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        # Evaluate the agent
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_rewards.append(avg_reward)
            print(f"Episode: {episode}, Avg. Reward: {avg_reward:.2f}")
            
        # Record the episode
        if episode % record_interval == 0:
            agent.save('./tracking', f"episode_{episode}.pth")
            print(f"Saved episode {episode} model")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_episodes, eval_interval), avg_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'SAC Learning Curve - {env_name}')
    plt.grid(True)
    plt.savefig(f"{figure_save_dir}/sac_{env_name}_learning_curve.png")
    plt.show()
    
    return episode_rewards, avg_rewards

# Function to render the trained policy
def render_policy(env_name, agent, max_steps=1000, save_path=''):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    
    frames = []
    total_reward = 0
    
    for step in range(max_steps):
        # Render the environment
        frame = env.render()
        frames.append(frame)
        
        # Select action
        action = agent.select_action(state, evaluate=True)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        state = next_state
        
        if done:
            break
    
    env.close()
    print(f"Total reward: {total_reward:.2f}")
    
    # Save as GIF
    display(save_frames_as_gif(frames, file_name=f"{env_name}.gif"))
    return frames