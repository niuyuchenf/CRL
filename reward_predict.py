import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, matrix_size, temperature=0.5, hard=False):
        super(GumbelSoftmaxLayer, self).__init__()
        self.hard = hard
        self.matrix_size = matrix_size
        self.temperature = temperature
        self.m, self.n = matrix_size
        self.logits = nn.Parameter(torch.randn(self.m, self.n, 2))

    def forward(self):
        if self.training:
            gumbels = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-10) + 1e-10)
            logits = (self.logits + gumbels) / self.temperature
        else:
            logits = self.logits / self.temperature
        
        y = F.softmax(logits, dim=-1)
        
        if self.hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            samples = (y_hard - y).detach() + y  
        else:
            samples = y  
        
        return samples, y


class FullConnectedLayers(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RewardNetwork(nn.Module):
    def __init__(self, matrix_size_state, matrix_size_action, temperature=1.0, 
                 output_dim=1, lambda_reg=1e-5, lambda_entropy=0.01, hard=False):
        super(RewardNetwork, self).__init__()
        self.gumbel_softmax_layer1 = GumbelSoftmaxLayer(matrix_size_state, temperature, hard)
        self.gumbel_softmax_layer2 = GumbelSoftmaxLayer(matrix_size_action, temperature, hard)
        self.n_state, _ = matrix_size_state
        self.n_action, _ = matrix_size_action
        self.fc_layers = FullConnectedLayers(
            self.n_state * matrix_size_state[1] + self.n_action * matrix_size_action[1], 
            output_dim=output_dim
        )
        self.lambda_reg = lambda_reg
        self.lambda_entropy = lambda_entropy  
        self.initial_temperature = temperature

    def forward(self, input_vector1, input_vector2):
        sampled_matrix1, softmax_probs1 = self.gumbel_softmax_layer1()
        sampled_matrix2, softmax_probs2 = self.gumbel_softmax_layer2()

        if self.gumbel_softmax_layer1.hard:
            binary_matrix1 = sampled_matrix1.argmax(dim=-1).float()
            binary_matrix2 = sampled_matrix2.argmax(dim=-1).float()
        else:
            binary_matrix1 = sampled_matrix1[:, :, 1]
            binary_matrix2 = sampled_matrix2[:, :, 1]

        input_vector1 = input_vector1.to(binary_matrix1.device)
        input_vector2 = input_vector2.to(binary_matrix2.device)
        
        if len(input_vector1.shape) == 1:
            input_vector1 = input_vector1.unsqueeze(0)
        if len(input_vector2.shape) == 1:
            input_vector2 = input_vector2.unsqueeze(0)

        batch_size = input_vector1.size(0)

        binary_matrix1 = binary_matrix1.unsqueeze(0).expand(batch_size, -1, -1)
        binary_matrix2 = binary_matrix2.unsqueeze(0).expand(batch_size, -1, -1)

        combined1 = binary_matrix1 * input_vector1.unsqueeze(2)
        combined2 = binary_matrix2 * input_vector2.unsqueeze(2)
        combined = torch.cat([combined1, combined2], dim=1)
        combined_flattened = combined.view(batch_size, -1)
        final_output = self.fc_layers(combined_flattened)

        return final_output, softmax_probs1, softmax_probs2

    
    def loss_reward_prediction(self, predicted_rewards, actual_trajectory_return, 
                              softmax_probs1, softmax_probs2, gamma=0.99):
       
        predicted_rewards = predicted_rewards.squeeze()
        if predicted_rewards.dim() == 0:
            predicted_rewards = predicted_rewards.unsqueeze(0)

        T = predicted_rewards.size(0)

        discount_factors = torch.tensor([gamma ** i for i in range(T)], 
                                       device=predicted_rewards.device)

        predicted_trajectory_return = torch.sum(predicted_rewards * discount_factors)

        reward_loss = (predicted_trajectory_return - actual_trajectory_return) ** 2

        sparsity_loss = softmax_probs1[:, :, 1].sum() + softmax_probs2[:, :, 1].sum()
        
        entropy1 = -(softmax_probs1 * torch.log(softmax_probs1 + 1e-10)).sum(dim=-1).mean()
        entropy2 = -(softmax_probs2 * torch.log(softmax_probs2 + 1e-10)).sum(dim=-1).mean()
        
        total_loss = (reward_loss + 
                     self.lambda_reg * sparsity_loss + 
                     self.lambda_entropy * (entropy1 + entropy2))
        
        return total_loss

    def anneal_temperature(self, epoch, total_epochs, min_temp=0.5):
        new_temp = max(min_temp, self.initial_temperature * (1 - epoch / total_epochs))
        self.gumbel_softmax_layer1.temperature = new_temp
        self.gumbel_softmax_layer2.temperature = new_temp
    
    def get_causal_matrices(self):
        self.eval()
        with torch.no_grad():
            _, probs1 = self.gumbel_softmax_layer1()
            _, probs2 = self.gumbel_softmax_layer2()
            binary1 = (probs1[:, :, 1] > 0.5).float()
            binary2 = (probs2[:, :, 1] > 0.5).float()
        return binary1, binary2

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))


def compute_discounted_return(rewards, gamma=0.99):

    T = len(rewards)
    gamma_powers = torch.pow(gamma, torch.arange(T, device=rewards.device, dtype=rewards.dtype))
    return torch.sum(rewards * gamma_powers)

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def reward_predict(trajectories, n_state_features, n_reward_features, temperature, 
                  n_action_features, num_epochs, lambda_reg, number):

    set_seed(98)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_matrix_size = (n_state_features, n_reward_features)
    action_matrix_size = (n_action_features, n_reward_features)
    net = RewardNetwork(
        state_matrix_size, 
        action_matrix_size, 
        temperature=temperature,
        output_dim=1, 
        lambda_reg=lambda_reg,
        lambda_entropy=0.01,  
        hard=False  
    )
    net.to(device)
    model_path = "./reward/reward_{}.pth".format(number)
    net.load_parameters(model_path, device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    
    gamma = 0.99
    train_trajectories = []
    for traj in trajectories:
        train_trajectories.append({
            'states': traj['states'].to(torch.float32).to(device),
            'actions': traj['actions'].to(torch.float32).to(device),
            'rewards': traj['rewards'].to(torch.float32).to(device),
            'return': compute_discounted_return(traj['rewards'].to(torch.float32).to(device), gamma)
        })
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        total_batch_loss = 0  
        epoch_loss = 0 
        
        for traj_idx, traj in enumerate(train_trajectories):
            states = traj['states']      # [T, state_dim]
            actions = traj['actions']    # [T, action_dim]
            trajectory_return = traj['return']  # scalar
            
            outputs, softmax_probs1, softmax_probs2 = net(states, actions)
            predicted_rewards = outputs.squeeze()  # [T]
            
            loss = net.loss_reward_prediction(
                predicted_rewards, 
                trajectory_return, 
                softmax_probs1, 
                softmax_probs2,
                gamma=gamma
            )
            
            total_batch_loss += loss
            epoch_loss += loss.item()
        
        average_batch_loss = total_batch_loss / len(train_trajectories)
        
        average_batch_loss.backward()
        
        optimizer.step()
        
        if epoch % 10 == 0:
            net.anneal_temperature(epoch, num_epochs, min_temp=0.6)
        
        scheduler.step()
        
    
    net.save_parameters(f"./reward/reward_{number}.pth")
