from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Charger le tokenizer et le modèle 
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Définir les phrases en anglais pour "turn right" et "turn left"
phrases_right = [
    "Turn right",
    "Go to the opposite of my left hand",
    "Go East",
    "Take a turn at the next intersection"
]

phrases_left = [
    "Turn left",
    "Hey ! Please go left man you're close",
    "Ok go to the opposite direction",
    "Go away then go left"
]

# Combiner toutes les phrases pour les encoder ensemble
all_phrases = phrases_right + phrases_left

# Tokenisation du texte
inputs = tokenizer(all_phrases, padding=True, truncation=True, return_tensors="pt")

# Encodage du texte
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Moyenne des embeddings de tokens pour chaque phrase

print("Embeddings shape:", embeddings.shape)
# Convertir les embeddings en numpy pour calculer la similarité cosinus
embeddings_np = embeddings.cpu().numpy()

# Reprendre les embeddings calculés pour "right" et "left"
embeddings_right = embeddings_np[:len(phrases_right)]  # Embeddings pour les phrases "right"
embeddings_left = embeddings_np[len(phrases_right):]  # Embeddings pour les phrases "left"

# Calculer les similarités cosinus pour chaque groupe
similarity_right = cosine_similarity(embeddings_right)
similarity_left = cosine_similarity(embeddings_left)

# Tracer la matrice de similarité pour les phrases "right-right"
plt.figure(figsize=(6, 6))
sns.heatmap(similarity_right, annot=True, cmap='coolwarm', xticklabels=phrases_right, yticklabels=phrases_right,
            cbar_kws={'label': 'Cosine Similarity'}, fmt='.4f', linewidths=0.5, square=True)
plt.title("Intra-group Similarity Matrix (Right-Right)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Tracer la matrice de similarité pour les phrases "left-left"
plt.figure(figsize=(6, 6))
sns.heatmap(similarity_left, annot=True, cmap='coolwarm', xticklabels=phrases_left, yticklabels=phrases_left,
            cbar_kws={'label': 'Cosine Similarity'}, fmt='.4f', linewidths=0.5, square=True)
plt.title("Intra-group Similarity Matrix (Left-Left)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
