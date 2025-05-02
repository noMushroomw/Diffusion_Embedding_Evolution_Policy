import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate expert trajectories
def generate_expert_trajectories(env, agent, num_trajectories=50, max_steps=1000):
    """
    Generate expert trajectories using the trained SAC agent and format them
    as an offline RL dataset simultaneously
    """
    trajectories = []
    
    # For offline RL dataset
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    
    for i in tqdm(range(num_trajectories), desc="Generating expert trajectories"):
        state, _ = env.reset()
        trajectory = []
        
        for step in range(max_steps):
            # Expert action
            action = agent.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store state-action pair in trajectory
            trajectory.append((state.tolist(), action.tolist()))
            
            # Store transition in offline RL dataset format
            all_states.append(state.tolist())
            all_actions.append(action.tolist())
            all_next_states.append(next_state.tolist())
            all_rewards.append(reward)
            all_dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        trajectories.append(trajectory)
        #print(f"Trajectory {i+1}: {len(trajectory)} steps")
    
    print(f"Generated {len(trajectories)} expert trajectories")
    # Create offline RL dataset
    offline_dataset = {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'next_states': np.array(all_next_states),
        'rewards': np.array(all_rewards),
        'dones': np.array(all_dones)
    }
    
    return trajectories, offline_dataset


# Create offline dataset in a format suitable for offline RL
def create_offline_dataset(trajectories):
    """Create offline dataset from trajectories"""
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            state, action = trajectory[i]
            next_state, next_action = trajectory[i + 1]
            
            # For the reward and done flag, we'll use a simple heuristic
            # In practice, you might want to run the environment with these states/actions
            # to get the exact reward, but this simpler approach avoids potential errors
            reward = 0.0  # Default reward
            done = False
            
            if i == len(trajectory) - 2:  # Last transition in trajectory
                done = True
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
    
    # Convert to numpy arrays
    dataset = {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'rewards': np.array(rewards),
        'dones': np.array(dones)
    }
    
    return dataset


# Generate embeddings for all data points
def generate_all_embeddings(dataset, sa_embedding):
    # Convert to tensors
    states = torch.FloatTensor(dataset['states']).to(device)
    actions = torch.FloatTensor(dataset['actions']).to(device)
    
    # Process in batches to avoid memory issues
    batch_size = 256
    sa_embeddings = []
    
    for i in range(0, len(states), batch_size):
        batch_states = states[i:i+batch_size]
        batch_actions = actions[i:i+batch_size]
        
        # Generate embeddings
        with torch.no_grad():
            batch_sa_embeddings = sa_embedding(batch_states, batch_actions)
        
        # Convert to numpy and save
        sa_embeddings.append(batch_sa_embeddings.cpu().numpy())
    
    # Concatenate all batches
    sa_embeddings = np.concatenate(sa_embeddings)
    
    return sa_embeddings


# Visualize embeddings with t-SNE
def visualize_embeddings(embeddings, title, n_samples=1000, save_dir='./figures'):
    # Sample embeddings if too many
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title(f"t-SNE visualization of {title}")
    plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()


# visualize embeddings evolution for all intermediate steps
def visualize_embeddings_evolution(embeddings_list, titles, file_name, n_samples=1000, save_dir='./figures'):
    """
    Visualize the evolution of embeddings over different training steps
    """
    embeddings2d_list = []
    for i, embeddings in enumerate(embeddings_list):
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings_sample = embeddings[indices]
        else:
            embeddings_sample = embeddings
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        embeddings2d_list.append(embeddings_2d)
    # Plotting
    
        # subplot for each embedding
    fig, axes = plt.subplots(len(embeddings_list), 1,
                             figsize=(10, 8 * len(embeddings_list)), sharex=True, sharey=True)
    for i, embeddings_2d in enumerate(embeddings2d_list):
        axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        axes[i].set_title(f"t-SNE visualization of episode {10 * (i+1)}")
        axes[i].set_xlabel("t-SNE 1")
        axes[i].set_ylabel("t-SNE 2")
    plt.savefig(os.path.join(save_dir, file_name))
    plt.show()        
