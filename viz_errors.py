import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model

# Charger les données
X_test = np.load('./data/X_test_normalized.npy')  # Données d'observations
y_test = np.load('./data/y_test_normalized.npy')  # Données de vérité terrain (trajectoires réelles)

# Paramètres
n_robots = 20  # Nombre de robots à afficher (modifier selon tes besoins)
checkpoint_path = './models/256_128_neur+forward+backwards+sharp+Roberta+NormalizedMinMSELoss/model_epoch_200.pth'  # Chemin du checkpoint

# Paramètres du modèle
input_size = X_test.shape[1]
fc_size1 = 256
fc_size2 = 128
output_size = y_test.shape[-1]

# Charger le modèle
model = load_model(checkpoint_path, input_size, fc_size1, fc_size2, output_size)

# Sélectionner aléatoirement des robots parmi les données
indices = np.random.choice(X_test.shape[0], size=n_robots, replace=False)
X_sample = X_test[indices]
y_sample = y_test[indices]

# Prédictions du modèle
with torch.no_grad():
    X_tensor = torch.tensor(X_sample, dtype=torch.float32)
    y_pred = model(X_tensor).numpy()

"""
X_sample is of shape : (n_robots, observation_size)
with observation = [v_r, theta_r, d_g, theta_g, l1, l2, ..., l40, emb1, emb2, ..., emb768]

y_sample is of shape : (n_robots, num_targets, output_size)
with output = [x1, y1, x2, y2, ..., x5, y5]
"""
for i in range(n_robots):
    plt.figure(figsize=(10, 6))
    for j in range(y_sample.shape[1] // 2):  # Pour chaque point de l'instruction (5 paires x, y)
        y_xreal = y_sample[i, j, ::2]  # Indices pairs pour les x réels (target points)
        y_yreal = y_sample[i, j, 1::2]  # Indices impairs pour les y réels (target points)
        
        plt.plot(y_xreal, y_yreal, 'go--', label=f'Robot {i+1} Vérité Terrain', markersize=5)

    # Prédictions (y_pred contient des [x1, y1, x2, y2, ...])
    y_xpred_i = y_pred[i, ::2]  # Indices pairs pour les x prédits
    y_ypred_i = y_pred[i, 1::2]  # Indices impairs pour les y prédits

    plt.plot(y_xpred_i, y_ypred_i, 'ro--', label=f'Robot {i+1} Prédiction', markersize=5)

    # Configuration du plot
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.gca().invert_yaxis()  # Inverser l'axe Y pour correspondre au système de coordonnées
    plt.title(f'Vérité terrain vs Prédiction du modèle pour {n_robots} robots')
    # plt.legend()
    plt.grid(True)
    plt.show()
