import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Gymnasium_envs
from transformers import AutoTokenizer, AutoModel
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
import pygame
from typing import Tuple
from tqdm import tqdm

from utils import load_model, get_embeddings

def c2p(cart):
        r = np.linalg.norm(cart)
        theta = np.arctan2(cart[1], cart[0])
        return np.array([r, theta])


def generate_turn_points(observation, embedding, model):
    # Concaténer les observations et l'embedding
    input_data = np.concatenate([observation.flatten(), embedding.flatten()])
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Ajouter une dimension pour le batch
    
    # Prédire les points de sortie
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).numpy()
    
    return output

# Fonction principale
def main(checkpoint_path, nlp_model, env_name="Navigation-v0", n_rays=40, max_steps=100):

    tokenizer = AutoTokenizer.from_pretrained(nlp_model)
    text_model = AutoModel.from_pretrained(nlp_model)
    embedding_size = text_model.config.hidden_size
    pygame.init()
    
     # Initialiser l'environnement
    env = gym.make(env_name, n_rays=n_rays, max_steps=max_steps, render_mode=None)
    
    # Charger le modèle MLP depuis le checkpoint
    input_size = env.observation_space.shape[0] + embedding_size 
    fc_size1 = 256
    fc_size2 = 128
    output_size = 10  # 5 points x et 5 points y
    
    mlp_model = load_model(checkpoint_path, input_size, fc_size1, fc_size2, output_size)
    
    # Boucle principale
    done = False
    # Load model
    kwargs = {'n_rays': 40, 'max_steps': 150}
    kwargs['render_mode'] = 'human'
    model_path = "./navigation/results/sac_Nav40Rays/models/rl_model_999984_steps.zip"
    env_path = "./navigation/results/sac_Nav40Rays/models/rl_model_vecnormalize_999984_steps.pkl"
    env = VecNormalize.load(env_path, make_vec_env(env_name, n_envs=1, env_kwargs=kwargs))
    model = SAC.load(model_path, device='cpu')
    observation = env.reset()
    
    objectives = []
    while 1:
        action, _states = model.predict(observation, deterministic=True)
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
            
        env.render()  # Affiche l'environnement

        # Vérifier si l'utilisateur appuie sur "C"
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                instructions = dict()
                for instruction in [
                    "Turn left.",
                    "Make a wide left turn.",
                    "Make a sharp left turn.",
                    "Turn right.",
                    "Make a wide right turn.",
                    "Make a sharp right turn.",
                    "Move forward.",
                    "Move backward."
                    ]:
                
                    print(f"Instruction: {instruction}")
                    # Obtenir l'embedding de l'instruction
                    embedding = get_embeddings(text_model, tokenizer, [instruction])[0]
                    
                    # Generate the output using the unnormalized observation
                    output = generate_turn_points(observation[0].flatten(), embedding.flatten(), mlp_model)
                    x_points = output[::2]
                    y_points = output[1::2]
                    x_robot, y_robot = env.envs[0].get_wrapper_attr('agent_pos')
                    
                    x_points += x_robot
                    y_points += y_robot
                    coordinates = list(zip(x_points, y_points))
                    objectives = np.array(coordinates[1:])
                    # Remove points that are outside the map
                    objectives = objectives[objectives[:, 0] > -10]
                    objectives = objectives[objectives[:, 0] < 10]
                    instructions[instruction] = coordinates

                # Set the coordinate list in the environment
                env.envs[0].unwrapped.set_coordinate_list(instructions)
                time.sleep(1)
                env.render()
                time.sleep(10)

                ############################
                # x_points_plot = x_robot + x_points
                # y_points_plot = y_robot + y_points
                # # Obtenir l'inertie (angle) du robot
                # inertia_angle = observation[0][1] 

                # # Plot simple avec matplotlib
                # plt.figure(figsize=(8, 8))
                # plt.plot(x_points_plot, y_points_plot, 'ro-', label="Trajectoire prédite")
                # plt.plot(x_robot, y_robot, 'go', markersize=10, label="Position du Robot")

                # # Ajouter des numéros aux points
                # for i, (x, y) in enumerate(zip(x_points_plot, y_points_plot)):
                #     plt.text(x, y, f'{i+1}', fontsize=12, ha='right')

                # # Dessiner une flèche pour l'inertie
                # arrow_length = 2
                
                # plt.arrow(x_robot, y_robot, arrow_length * np.cos(inertia_angle), arrow_length * np.sin(inertia_angle),
                #         head_width=0.5, head_length=0.5, fc='blue', ec='blue', label="Inertie")

                # # Configurer le plot
                # plt.xlabel('Position X')
                # plt.ylabel('Position Y')
                # plt.title('Trajectoire du Robot avec Inertie')
                # plt.grid(True)
                # plt.legend()
                # plt.xlim([-20, 20])
                # plt.ylim([-20, 20])
                # # Invert Y axis to match pygame's coordinate system
                # plt.gca().invert_yaxis()
                # plt.show()
                ############################

                
                

    env.close()

# Exemple d'appel à la fonction principale
if __name__ == '__main__':
    checkpoint_path = './models/256_128_neur+forward+backwards+sharp+Roberta+NormalizedMinMSELoss_GENERALIZATION/model_epoch_200.pth'
    model_name = "roberta-base"
    main(checkpoint_path, model_name)