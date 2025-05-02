import os
import torch
import pickle
from scripts.sac import SAC
from scripts.action_embedding_net import StateActionEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained SAC agent
def load_agent(env, path):
    """
    Load a trained SAC agent from the specified path.
    
    Args:
        env: The environment to load the agent for.
        path: The path to the trained agent's weights.
        
    Returns:
        agent: The loaded SAC agent.
    """
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        lr=3e-4,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        auto_entropy_tuning=True
    )
    
    # Load trained weights
    agent.load(path)
    print("Loaded trained SAC agent")
    
    return agent


# Load the expert trajectories
def load_trajectories(filename="./expert_data/expert_trajectories.pkl"):
    """Load expert trajectories from pickle file"""
    with open(filename, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


# Load the offline dataset
def load_offline_dataset(filename="./expert_data/halfcheetah_expert_dataset.pkl"):
    """Load offline dataset from pickle file"""
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def load_embedding_models(state_dim, action_dim, directory="./models"):
    """Load trained embedding models"""
    sa_embedding = StateActionEmbedding(state_dim, action_dim).to(device)
    sa_embedding.load_state_dict(torch.load(os.path.join(directory, "state_action_embedding.pt")))
    
    print(f"Loaded embedding models from {directory}")
    
    return sa_embedding


# Function to get embedding for a new state-action pair
def get_state_action_embedding(state, action, model):
    """Get embedding for a new state-action pair"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).to(device)
        action_tensor = torch.FloatTensor(action).to(device)
        
        # Handle single state-action pair
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
        
        embedding = model(state_tensor, action_tensor)
        
        return embedding.cpu().numpy()