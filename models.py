import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, fc_size1, fc_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, fc_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size1, fc_size2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(fc_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
    
class MinMSELoss(nn.Module):
    def __init__(self):
        super(MinMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Compute MSE without reduction
    
    def forward(self, outputs, target_set):
        # Outputs shape: (1, output_size), Target set shape: (1, num_targets, output_size)
        mse_losses = self.mse_loss(outputs.unsqueeze(1), target_set)  # Shape: (1, num_targets, output_size)
        mse_losses = mse_losses.mean(dim=-1)  # Mean over output_size, resulting in shape: (1, num_targets)
        min_mse_loss = mse_losses.min(dim=1)[0]  # Minimum MSE over targets, shape: (1,)
        return min_mse_loss.mean() # Mean over batch

class NormalizedMinMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(NormalizedMinMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Compute MSE without reduction
        self.epsilon = epsilon  # Small value to prevent division by zero

    
    def forward(self, outputs, target_set):
        # Outputs shape: (batch_size, output_size)
        # Target set shape: (batch_size, num_targets, output_size)
        batch_size, num_targets, output_size = target_set.shape

        # Calculate MSE losses without reduction
        mse_losses = self.mse_loss(outputs.unsqueeze(1), target_set)  # Shape: (batch_size, num_targets, output_size)
        mse_losses = mse_losses.mean(dim=-1)  # Mean over output_size, resulting in shape: (batch_size, num_targets)

        # Calculate total distances for each target in the target set
        num_points = output_size // 2

        # Reshape target_set to (batch_size, num_targets, num_points, 2)
        target_set_points = target_set.view(batch_size, num_targets, num_points, 2)

        # Calculate distances between consecutive points
        distances = torch.norm(
            target_set_points[:, :, 1:, :] - target_set_points[:, :, :-1, :],  # Differences between consecutive points
            dim=-1
        )  # Shape: (batch_size, num_targets, num_points - 1)

        # Sum distances to get total distance per target
        total_distances = distances.sum(dim=-1)  # Shape: (batch_size, num_targets)
        total_distances = total_distances + self.epsilon

        # Normalize mse_losses by total_distances
        normalized_losses = mse_losses / total_distances

        # Find minimum loss over targets
        min_normalized_loss = normalized_losses.min(dim=1)[0]  # Shape: (batch_size,)

        return min_normalized_loss.mean()