import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for training the embedding networks
class ExpertDataset(Dataset):
    def __init__(self, dataset, negative_samples=5):
        self.states = torch.FloatTensor(dataset['states']).to(device)
        self.actions = torch.FloatTensor(dataset['actions']).to(device)
        self.next_states = torch.FloatTensor(dataset['next_states']).to(device)
        self.rewards = torch.FloatTensor(dataset['rewards']).to(device)
        self.dones = torch.FloatTensor(dataset['dones']).to(device)
        
        self.num_samples = len(self.states)
        self.negative_samples = negative_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        # Create positive sample - actual state-action pair
        pos_state = state
        pos_action = action
        
        # Create negative samples - random actions with same state
        neg_indices = torch.randint(0, self.num_samples, (self.negative_samples,))
        neg_actions = self.actions[neg_indices]
        
        return state, action, neg_actions



# State-Action Embedding Network
class StateActionEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim=64, hidden_dim=128):
        super(StateActionEmbedding, self).__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, embedding_dim),
            nn.Tanh()  # Tanh to keep embeddings between -1 and 1
        )
    
    def encode_state(self, state):
        return self.state_encoder(state)
    
    def encode_action(self, action):
        return self.action_encoder(action)
    
    def forward(self, state, action):
        state_features = self.encode_state(state)
        action_features = self.encode_action(action)
        combined = torch.cat([state_features, action_features], dim=-1)
        embedding = self.combined_encoder(combined)
        return embedding

# Action-only Embedding Network
class ActionEmbedding(nn.Module):
    def __init__(self, action_dim, embedding_dim=32, hidden_dim=64):
        super(ActionEmbedding, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Tanh to keep embeddings between -1 and 1
        )
    
    def forward(self, action):
        return self.encoder(action)

# Contrastive loss for embeddings
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negatives):
        # Compute similarity between anchor and positive
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        
        # Compute similarities between anchor and negatives
        neg_sims = []
        for i in range(negatives.size(1)):
            neg_sim = F.cosine_similarity(anchor, negatives[:, i], dim=-1)
            neg_sims.append(neg_sim)
        
        # Stack negative similarities
        neg_sims = torch.stack(neg_sims, dim=-1)
        
        # Compute triplet loss
        # We want pos_sim to be higher than all neg_sims by at least margin
        losses = []
        for i in range(neg_sims.size(1)):
            loss = F.relu(self.margin - pos_sim + neg_sims[:, i])
            losses.append(loss)
        
        # Return mean loss
        return torch.mean(torch.stack(losses))

# Train the embedding networks
def train_embeddings(dataset, state_dim, action_dim, epochs=50, batch_size=256, lr=1e-3):
    # Create dataset and dataloader
    expert_dataset = ExpertDataset(dataset)
    dataloader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
    
    # Create models
    sa_embedding = StateActionEmbedding(state_dim, action_dim).to(device)
    a_embedding = ActionEmbedding(action_dim).to(device)
    
    # Create optimizers
    sa_optimizer = optim.Adam(sa_embedding.parameters(), lr=lr)
    a_optimizer = optim.Adam(a_embedding.parameters(), lr=lr)
    
    # Create loss functions
    sa_criterion = ContrastiveLoss()
    a_criterion = ContrastiveLoss()
    
    # Training loop
    sa_losses = []
    a_losses = []
    
    for epoch in range(epochs):
        sa_epoch_loss = 0
        a_epoch_loss = 0
        num_batches = 0
        
        for state, action, neg_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # State-action embedding loss
            sa_anchor = sa_embedding(state, action)
            
            # Random negative state-action pairs (using same state but different actions)
            sa_positives = sa_anchor  # Use the anchor itself as positive (for simplicity)
            sa_negatives = torch.stack([sa_embedding(state, neg_action) for neg_action in neg_actions.unbind(dim=1)], dim=1)
            
            sa_loss = sa_criterion(sa_anchor, sa_positives, sa_negatives)
            
            sa_optimizer.zero_grad()
            sa_loss.backward()
            sa_optimizer.step()
            
            # Action embedding loss
            a_anchor = a_embedding(action)
            
            # Random negative actions
            a_positives = a_anchor  # Use the anchor itself as positive (for simplicity)
            a_negatives = torch.stack([a_embedding(neg_action) for neg_action in neg_actions.unbind(dim=1)], dim=1)
            
            a_loss = a_criterion(a_anchor, a_positives, a_negatives)
            
            a_optimizer.zero_grad()
            a_loss.backward()
            a_optimizer.step()
            
            # Update metrics
            sa_epoch_loss += sa_loss.item()
            a_epoch_loss += a_loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        sa_avg_loss = sa_epoch_loss / num_batches
        a_avg_loss = a_epoch_loss / num_batches
        
        sa_losses.append(sa_avg_loss)
        a_losses.append(a_avg_loss)
        
        #print(f"Epoch {epoch+1}/{epochs}, SA Loss: {sa_avg_loss:.4f}, A Loss: {a_avg_loss:.4f}")
        # update progress bar every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, SA Loss: {sa_avg_loss:.4f}, A Loss: {a_avg_loss:.4f}")
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sa_losses)
    plt.title('State-Action Embedding Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(a_losses)
    plt.title('Action Embedding Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('embedding_training_losses.png')
    plt.show()
    
    return sa_embedding, a_embedding, sa_losses, a_losses