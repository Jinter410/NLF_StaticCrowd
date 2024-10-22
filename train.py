import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import MLP, MinMSELoss, NormalizedMinMSELoss
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

run_name = "256_128_neur+forward+backwards+sharp+Roberta+NormalizedMinMSELoss"
train_percentage = 0.7
n_epochs = 200
save_freq = 10
fc_size1 = 256
fc_size2 = 128

X = np.load('./data/X_normalized.npy')
y = np.load('./data/y_normalized.npy')

indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(train_percentage * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Params
input_size = X.shape[1]
output_size = y.shape[-1]
model = MLP(input_size, fc_size1, fc_size2, output_size)
criterion = NormalizedMinMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# TensorBoard
log_dir = f'./tensorboard_logs/{run_name}'
writer = SummaryWriter(log_dir)

# Checkpoints
checkpoint_dir = f'./models/{run_name}'
os.makedirs(checkpoint_dir, exist_ok=True)

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(loader.dataset)

train_losses = []
val_losses = []
progress_bar = tqdm(total=n_epochs, desc="Training Progress")

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    # Tqdm
    progress_bar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
    progress_bar.update(1)
    
    # Sauvegarder un checkpoint tous les 10 epochs
    if (epoch + 1) % save_freq == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

writer.close()
progress_bar.close()

# Plots
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss')
plt.show()