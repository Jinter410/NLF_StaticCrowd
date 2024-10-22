from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Tuple
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from matplotlib import pyplot as plt
import numpy as np
import torch
import Gymnasium_envs
import pygame
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils import generate_one, get_embeddings

INSTRUCTIONS = {
    "left": [
        "Turn left.",
        "Rotate left.",
        "Take a left turn.",
        "Move leftward.",
        "Steer to the left.",
        "Swing left.",
        "Adjust course to the left.",
        "Head to the left.",
        "Shift to the left.",
        "Angle left."
    ],
    "right": [
        "Turn right.",
        "Rotate right.",
        "Take a right turn.",
        "Move rightward.",
        "Steer to the right.",
        "Swing right.",
        "Adjust course to the right.",
        "Head to the right.",
        "Shift to the right.",
        "Angle right."
    ],
    "wide_left": [
        "Make a wide left turn.",
        "Turn left with a wide arc.",
        "Take a broad left curve.",
        "Swing widely to the left.",
        "Steer left in a wide turn.",
        "Make a large left turn.",
        "Curve left with a wide path.",
    ],
    "sharp_left": [
        "Make a sharp left turn.",
        "Turn left sharply.",
        "Take a quick left.",
        "Swing abruptly to the left.",
        "Steer left in a sharp turn.",
        "Make a tight left turn.",
        "Pivot sharply to the left."
    ],
    "wide_right": [
        "Make a wide right turn.",
        "Turn right with a wide arc.",
        "Take a broad right curve.",
        "Swing widely to the right.",
        "Steer right in a wide turn.",
        "Make a large right turn.",
        "Curve right with a wide path.",
    ],
    "sharp_right": [
        "Make a sharp right turn.",
        "Turn right sharply.",
        "Take a quick right.",
        "Swing abruptly to the right.",
        "Steer right in a sharp turn.",
        "Make a tight right turn.",
        "Pivot sharply to the right."
    ],
    "forward": [
        "Move forward.",
        "Go straight.",
        "Proceed straight ahead.",
        "Advance forward.",
        "Head straight.",
        "Continue forward.",
        "Move ahead.",
        "Keep going straight.",
        "Walk straight ahead.",
        "Progress forward."
    ],
    "backward": [
        "Move backward.",
        "Go back.",
        "Proceed backward.",
        "Advance backward.",
        "Head backward.",
        "Continue backward.",
        "Move back.",
        "Keep going backward.",
        "Walk backward.",
        "Progress backward."
    ]
}

PARAMETERS = {
    'right': {
        'radius_min': 2,
        'radius_max': 15,
        'angle_min': 70,
        'angle_max': 110,
        'strength_min': 0.5,
        'strength_max': 2
    },
    'left': {
        'radius_min': 2,
        'radius_max': 15,
        'angle_min': 70,
        'angle_max': 110,
        'strength_min': 0.5,
        'strength_max': 2
    },
    'sharp_right': {
        'radius_min': 2,
        'radius_max': 7,
        'angle_min': 90,
        'angle_max': 120,
        'strength_min': 0.5,
        'strength_max': 1
    },
    'wide_right': {
        'radius_min': 7,
        'radius_max': 15,
        'angle_min': 70,
        'angle_max': 100,
        'strength_min': 0.5,
        'strength_max': 2
    },
    'sharp_left': {
        'radius_min': 2,
        'radius_max': 7,
        'angle_min': 90,
        'angle_max': 120,
        'strength_min': 0.5,
        'strength_max': 1
    },
    'wide_left': {
        'radius_min': 7,
        'radius_max': 15,
        'angle_min': 70,
        'angle_max': 100,
        'strength_min': 0.5,
        'strength_max': 2
    },
    'forward': {
        'length_min': 5,
        'length_max': 20
    },
    'backward': {
        'length_min': 5,
        'length_max': 20
    }
}

NUM_SAMPLES_PER_INSTRUCTION = 100

def generate_sample(n_samples, env: gym.Env, robot_x: float, robot_y: float, inertia_angle: float, how: str, disc_output: int, **kwargs):
    samples = np.zeros((n_samples, disc_output, 2))
    for i in range(n_samples):
        x_rot, y_rot, radius, angle = generate_one(robot_x, robot_y, how, inertia_angle, **kwargs)
        # If the turn is out of bounds, regenerate
        half_width = env.get_wrapper_attr('WIDTH') / 2
        half_height = env.get_wrapper_attr('HEIGHT') / 2
        phs = env.get_wrapper_attr('PHS')

        while np.any(x_rot < -half_width + phs) or np.any(x_rot > half_width - phs) or \
            np.any(y_rot < -half_height + phs) or np.any(y_rot > half_height - phs):
            x_rot, y_rot, radius, angle = generate_one(robot_x, robot_y, how, inertia_angle,**kwargs)
        
        # Scattering
        indices = np.linspace(0, len(x_rot) - 1, disc_output, dtype=int)
        x_rot = x_rot[indices]
        y_rot = y_rot[indices]

        samples[i] = np.column_stack((x_rot, y_rot))
    return samples

def generate(how, model, tokenizer, disc_output = 5, n_rays=40, n_crowd=4, interceptor_percentage = 0.5, max_steps = 100, n_data =100, render_mode=None) -> Tuple[np.ndarray, np.ndarray]:
    pygame.init()

    env_path = "./navigation/results/sac_Nav40Rays/models/rl_model_vecnormalize_999984_steps.pkl"

    env_name = "Navigation-v0"
    if env_name == "Navigation-v0":
        kwargs = {'n_rays': n_rays, 'max_steps': max_steps, 'render_mode': render_mode}
    else:
        kwargs = {'n_rays': n_rays, 'n_crowd': n_crowd, 'interceptor_percentage': interceptor_percentage, 'max_steps': max_steps, 'render_mode': render_mode}
 

    env = VecNormalize.load(env_path, make_vec_env(env_name, n_envs=1, env_kwargs=kwargs))
    sentences = INSTRUCTIONS[how]
    
    embeddings = np.zeros((len(sentences), 768))
    for i,sentence in enumerate(sentences):
        embedding_i = get_embeddings(model, tokenizer, [sentence])
        embeddings[i] = embedding_i

    observation_size = env.observation_space.shape[0] + embeddings.shape[1]
    rot_size = disc_output * 2
    X = np.zeros((n_data, observation_size))
    y = np.zeros((n_data, NUM_SAMPLES_PER_INSTRUCTION, rot_size))

    # Retrieve parameters for the current instruction
    instruction_params = PARAMETERS.get(how, {})

    n_steps = np.random.randint(2, 5)
    for _ in range(n_data):
        emb_i = np.random.choice(embeddings.shape[0], 1)
        r_emb = embeddings[emb_i]

        observation = env.reset()
        for i in range(n_steps):
            action = [env.action_space.sample()]
            observation, reward, truncated, info = env.step(action)
            
        # Unnormalize only for the generate_sample function
        unnormalized_observation = env.unnormalize_obs(observation)
        inertia_angle = unnormalized_observation[0][1]
        robot_x, robot_y = env.envs[0].get_wrapper_attr('agent_pos')

        # Generate real-world coordinates with unnormalized values
        samples_i = generate_sample(NUM_SAMPLES_PER_INSTRUCTION, env.envs[0], robot_x, robot_y, inertia_angle, how, disc_output, **instruction_params)

        #############################
        # plt.plot(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')
        # plt.plot(robot_x, robot_y, 'go', markersize=10)
        # plt.arrow(robot_x, robot_y, 2 * np.cos(inertia_angle), 2 * np.sin(inertia_angle),
        #         head_width=0.5, head_length=0.5, fc='blue', ec='blue')

        # # Annoter chaque point avec son numéro
        # for idx, (x, y) in enumerate(zip(x_rot, y_rot)):
        #     plt.text(x, y, str(idx + 1), fontsize=12, color='red', ha='center', va='center')

        # plt.axis([-env.WIDTH, env.WIDTH, -env.HEIGHT, env.HEIGHT])
        # # # Invert Y axis to match pygame's coordinate system
        # # plt.gca().invert_yaxis()
        # plt.show()
        # quit()
        #############################
        
        X[_] = np.concatenate([observation[0].flatten(), r_emb.flatten()])
        # zipped_points = np.array([coord for pair in zip(x_rot, y_rot) for coord in pair])
        y[_] = samples_i.reshape(NUM_SAMPLES_PER_INSTRUCTION, -1)
        n_steps = np.random.randint(2, 5)
        
    env.close()
    return X, y

def generate_wrapper(how, disc_output, n_rays, max_steps, n_data):
    # Initialize the model and tokenizer within each process
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Call the generate function with the initialized model and tokenizer
    X_how, y_how = generate(how, model, tokenizer, disc_output=disc_output, n_rays=n_rays, max_steps=max_steps, n_data=n_data)
    return X_how, y_how

if __name__ == "__main__":
    instruction_keys = ['sharp_right', 'wide_right', 'sharp_left', 'wide_left', 'forward', 'backward']
    X_list = []
    y_list = []
    
    # Parameters
    disc_output = 5
    n_rays = 40
    max_steps = 10
    n_data = 2000  # Number of data samples per instruction

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(generate_wrapper, how, disc_output, n_rays, max_steps, n_data)
            for how in instruction_keys
        ]

        # Collect the results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            X_how, y_how = future.result()
            X_list.append(X_how)
            y_list.append(y_how)

    # Concatenate the results
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    # Save the datasets
    np.save("./data/X_normalized.npy", X)
    np.save("./data/y_normalized.npy", y)