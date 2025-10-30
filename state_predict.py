import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
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

class StateNetwork(nn.Module):
    def __init__(self, matrix_size_state, matrix_size_action, temperature=1.0, 
                 output_dim=10, lambda_reg=1e-5, lambda_entropy=0.01, hard=False):
        super(StateNetwork, self).__init__()
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

    def loss_state_prediction(self, predicted_state, actual_state, softmax_probs1, softmax_probs2):
        state_loss = F.mse_loss(predicted_state, actual_state)
        sparsity_loss = softmax_probs1[:, :, 1].sum() + softmax_probs2[:, :, 1].sum()
        
        entropy1 = -(softmax_probs1 * torch.log(softmax_probs1 + 1e-10)).sum(dim=-1).mean()
        entropy2 = -(softmax_probs2 * torch.log(softmax_probs2 + 1e-10)).sum(dim=-1).mean()
        
        total_loss = state_loss + self.lambda_reg * sparsity_loss + self.lambda_entropy * (entropy1 + entropy2)
        
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

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def next_state_predict(state_dim, action_dim, temperature, lambda_reg, state_features, 
                       action_features, next_state_features, batch_size, num_epochs, number):
    
    set_seed(98)
    total_indices = list(range(len(state_features)))
    random.shuffle(total_indices)
    train_indices = total_indices[:50] 
    
    train_states = state_features[train_indices]
    train_actions = action_features[train_indices]
    train_next_states = next_state_features[train_indices]
    
    if len(train_next_states.shape) == 1:
        train_next_states = train_next_states.view(-1, 1)
    
    matrix_size_state = (state_dim, state_dim)
    matrix_size_action = (action_dim, state_dim)
    model = StateNetwork(
        matrix_size_state, 
        matrix_size_action, 
        temperature=temperature,
        output_dim=state_dim,
        lambda_reg=lambda_reg,
        lambda_entropy=0.01,  
        hard=False 
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_path = "./next_state/state_{}.pth".format(number)
    model.load_parameters(model_path, device)
            
    train_states = train_states.to(device)
    train_actions = train_actions.to(device)
    train_next_states = train_next_states.to(device)
    
    train_dataset = TensorDataset(train_states, train_actions, train_next_states)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        
        for state_batch, action_batch, next_state_batch in train_dataloader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                predicted_next_states, softmax_probs1, softmax_probs2 = model(state_batch, action_batch)
                loss = model.loss_state_prediction(predicted_next_states, next_state_batch, 
                                                   softmax_probs1, softmax_probs2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        
        if epoch % 10 == 0:
            model.anneal_temperature(epoch, num_epochs, min_temp=0.5)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_parameters("./next_state/state_{}.pth".format(number))
