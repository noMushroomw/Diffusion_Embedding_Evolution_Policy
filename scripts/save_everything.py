import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to save frames as GIF
def save_frames_as_gif(frames, filename='halfcheetah.gif', path='./figures'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    if not os.path.exists(path):
        os.makedirs(path)
    anim.save(os.path.join(path, filename), writer='pillow', fps=30)
    return HTML(anim.to_jshtml())


# Save trajectories and offline dataset
def save_data(trajectories, offline_dataset, save_dir='./',trajectory_filename="expert_trajectories.pkl", dataset_filename="halfcheetah_expert_dataset.pkl"):
    """Save trajectories and offline dataset to pickle files"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_trajectory_filename = os.path.join(save_dir, trajectory_filename)
    save_dataset_filename = os.path.join(save_dir, dataset_filename)
    # Save trajectories
    with open(save_trajectory_filename, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_trajectory_filename}")
    
    # Save offline dataset
    with open(save_dataset_filename, 'wb') as f:
        pickle.dump(offline_dataset, f)
    print(f"Saved offline dataset with {len(offline_dataset['states'])} transitions to {save_dataset_filename}")
    
    
    # Save offline dataset
def save_offline_dataset(dataset, save_dir='./', filename="halfcheetah_expert_dataset.pkl"):
    """Save offline dataset"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = os.path.join(save_dir, filename)
    with open(save_filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved offline dataset with {len(dataset['states'])} transitions to {save_filename}")
    
    
# Save trained embedding models
def save_embedding_models(sa_embedding, a_embedding, directory="./models"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(sa_embedding.state_dict(), os.path.join(directory, "state_action_embedding.pt"))
    torch.save(a_embedding.state_dict(), os.path.join(directory, "action_embedding.pt"))
    
    print(f"Saved embedding models to {directory}")
    
    
# Save embeddings to file
def save_embeddings(sa_embeddings, a_embeddings, directory="./embeddings", prefix=""):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save as numpy arrays
    np.save(os.path.join(directory, prefix + "state_action_embeddings.npy"), sa_embeddings)
    np.save(os.path.join(directory, prefix + "action_embeddings.npy"), a_embeddings)
    
    print(f"Saved embeddings to {directory}")