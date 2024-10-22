# transformer_pipeline_with_visualization.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModel
import math
import time
import gymnasium as gym
import pygame
import Gymnasium_envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

# ================================
# Custom Loss Function
# ================================
class MinMSELoss(nn.Module):
    def __init__(self):
        super(MinMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Compute MSE without reduction

    def forward(self, outputs, target_set):
        # Outputs shape: (batch_size, output_size)
        # Target set shape: (batch_size, num_samples_per_instruction, output_size)
        mse_losses = self.mse_loss(outputs.unsqueeze(1), target_set)  # Shape: (batch_size, num_samples_per_instruction, output_size)
        mse_losses = mse_losses.mean(dim=-1)  # Mean over output_size, resulting in shape: (batch_size, num_samples_per_instruction)
        min_mse_loss = mse_losses.median(dim=1)[0]  # Median MSE over targets, shape: (batch_size,)
        return min_mse_loss.mean()  # Mean over batch

# ================================
# Positional Encoding for Transformer
# ================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ================================
# Transformer Model Definition
# ================================
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = input_size  # Set d_model to input_size for simplicity

        self.positional_encoder = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(self.d_model, self.output_size)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        src = self.positional_encoder(src)
        src = src.transpose(0, 1)  # Transformer expects input of shape (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Aggregate over sequence length
        output = self.decoder(output)
        return output

# ================================
# Training the Transformer Model
# ================================
def train_transformer_model():
    # Data Loading
    X = np.load('./data/X.npy')  # Shape: (num_samples, observation_size)
    y = np.load('./data/y.npy')  # Shape: (num_samples, num_samples_per_instruction, output_size)

    # Data Preparation
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, observation_size)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Shape: (num_samples, num_samples_per_instruction, output_size)

    # Reshape X_tensor to (num_samples, seq_len, input_size)
    # Here, seq_len = 1
    X_tensor = X_tensor.unsqueeze(1)  # Shape: (num_samples, 1, observation_size)

    # Dataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and validation sets
    train_percentage = 0.7
    train_size = int(train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Parameters
    input_size = X_tensor.size(-1)  # observation_size
    output_size = y_tensor.size(-1)  # output_size
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    # Initialize Model, Loss Function, Optimizer
    model = TransformerModel(input_size, output_size, nhead, num_layers, dim_feedforward, dropout)
    criterion = MinMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training Loop
    n_epochs = 50
    train_losses = []
    val_losses = []
    progress_bar = tqdm(total=n_epochs, desc="Training Progress")

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape: (batch_size, output_size)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        progress_bar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        progress_bar.update(1)

    progress_bar.close()

    # Plot Loss Curves
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.show()

    # Save the trained model
    model_save_path = './transformer_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Transformer model saved to {model_save_path}")

    return model, input_size, output_size

# ================================
# Visualization in Simulation Environment
# ================================
def visualize_in_simulation(model, input_size, output_size):
    # Function to load the transformer model (if needed)
    def load_transformer_model(checkpoint_path, input_size, output_size, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        model = TransformerModel(input_size, output_size, nhead, num_layers, dim_feedforward, dropout)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model

    # Utility functions
    def c2p(cart):
        r = np.linalg.norm(cart)
        theta = np.arctan2(cart[1], cart[0])
        return np.array([r, theta])

    def get_embeddings(model, tokenizer, sentences):
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def generate_turn_points(observation, embedding, model):
        # Concatenate the unnormalized observation and the embedding
        input_data = np.concatenate([observation.flatten(), embedding.flatten()])
        # Reshape to (batch_size, seq_len, input_size)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, input_size)

        # Generate output
        with torch.no_grad():
            output = model(input_tensor).squeeze(0).numpy()

        return output

    # Main visualization function
    def main():
        # Load the NLP model and tokenizer
        nlp_model_name = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(nlp_model_name)
        text_model = AutoModel.from_pretrained(nlp_model_name)
        embedding_size = text_model.config.hidden_size

        pygame.init()

        # Initialize the environment
        env_name = "Navigation-v0"
        n_rays = 40
        max_steps = 150
        kwargs = {'n_rays': n_rays, 'max_steps': max_steps, 'render_mode': None}
        env = gym.make(env_name, **kwargs)

        # Adjust input size if necessary
        input_size = env.observation_space.shape[0] + embedding_size  # Adjusted input size

        # Load the SAC model
        kwargs['render_mode'] = 'human'
        model_path = "./navigation/results/sac_Nav40Rays/models/rl_model_999984_steps.zip"
        env_path = "./navigation/results/sac_Nav40Rays/models/rl_model_vecnormalize_999984_steps.pkl"
        env = VecNormalize.load(env_path, make_vec_env(env_name, n_envs=1, env_kwargs=kwargs))
        sac_model = SAC.load(model_path, device='cpu')
        observation = env.reset()

        objectives = []
        while True:
            action, _states = sac_model.predict(observation, deterministic=True)
            # After receiving the observation from env.step()
            observation, reward, done, info = env.step(action)
            if done:
                env.envs[0].unwrapped.set_coordinate_list([])


            if len(objectives) > 0:
                curr_objective = objectives[0].copy()

                if (np.linalg.norm(env.envs[0].get_wrapper_attr('agent_pos') - curr_objective) < 0.4):
                    objectives = objectives[1:]

                curr_objective -= env.envs[0].get_wrapper_attr('agent_pos')

                curr_objective_polar = c2p(curr_objective)

                # Unnormalize the observation
                unnormalized_obs = env.unnormalize_obs(observation)

                # Inject the unnormalized waypoint into the unnormalized observation
                unnormalized_obs[0][2:4] = curr_objective_polar

                # Re-normalize the observation
                observation = env.normalize_obs(unnormalized_obs)
                # time.sleep(1)

            env.render()  # Render the environment

            # Check if the user presses "C"
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    instructions = [
                        "Turn left.",
                        "Make a wide left turn.",
                        "Make a sharp left turn.",
                        "Make a wide right turn.",
                        "Make a sharp right turn.",
                        "Turn right.",
                        "Move forward.",
                        "Move backwards."
                        ]
                    for instruction in instructions:
                        print(instruction)
                        # Obtain the embedding of the instruction
                        embedding = get_embeddings(text_model, tokenizer, [instruction])[0]

                        # Generate the output using the unnormalized observation and transformer model
                        output = generate_turn_points(observation[0], embedding, model)
                        x_points = output[::2]
                        y_points = output[1::2]
                        x_robot, y_robot = env.envs[0].get_wrapper_attr('agent_pos')

                        x_points += x_robot
                        y_points += y_robot
                        coordinates = list(zip(x_points, y_points))
                        objectives = np.array(coordinates[1:])
                        # Remove points that are outside the map
                        objectives = objectives[(objectives[:, 0] > -10) & (objectives[:, 0] < 10)]
                        objectives = objectives[(objectives[:, 1] > -10) & (objectives[:, 1] < 10)]

                        # Set the coordinate list in the environment
                        env.envs[0].unwrapped.set_coordinate_list(coordinates)
                        time.sleep(1)
                        env.render()
                        time.sleep(5)

        env.close()

    # Call the main visualization function
    main()

# ================================
# Main Execution
# ================================
if __name__ == '__main__':
    # Train the transformer model and save it
    model, input_size, output_size = train_transformer_model()

    # Optionally, if you want to load the model from the saved file instead of the trained model in memory
    # Data Loading
    # X = np.load('./data/X.npy')  # Shape: (num_samples, observation_size)
    # y = np.load('./data/y.npy')  # Shape: (num_samples, num_samples_per_instruction, output_size)

    # # Data Preparation
    # # Convert to tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)  # Shape: (num_samples, observation_size)
    # y_tensor = torch.tensor(y, dtype=torch.float32)  # Shape: (num_samples, num_samples_per_instruction, output_size)

    # # Reshape X_tensor to (num_samples, seq_len, input_size)
    # # Here, seq_len = 1
    # X_tensor = X_tensor.unsqueeze(1)  # Shape: (num_samples, 1, observation_size)

    # # Dataset and DataLoader
    # dataset = TensorDataset(X_tensor, y_tensor)

    # # Split into train and validation sets
    # train_percentage = 0.7
    # train_size = int(train_percentage * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # batch_size = 128
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # Model Parameters
    # input_size = X_tensor.size(-1)  # observation_size
    # output_size = y_tensor.size(-1)  # output_size
    # model_save_path = './transformer_model.pth'
    # model = TransformerModel(input_size, output_size, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1)
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()

    # Visualize the model in the simulation environment
    visualize_in_simulation(model, input_size, output_size)
