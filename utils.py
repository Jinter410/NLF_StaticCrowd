import numpy as np
import pygame
import torch

from models import MLP

def load_model(checkpoint_path, input_size,  fc_size1, fc_size2, output_size):
    model = MLP(input_size,  fc_size1, fc_size2, output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def manual_action() -> tuple:
    keys = pygame.key.get_pressed()
    if not np.any(keys):
        action = (0, 0)
    else:
        linear_velocity = 1
        angle = None
        
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            angle = -np.pi / 4 
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            angle = -3 * np.pi / 4  
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            angle = np.pi / 4  
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            angle = 3 * np.pi / 4  
        elif keys[pygame.K_UP]:
            angle = -np.pi / 2 
        elif keys[pygame.K_DOWN]:
            angle = np.pi / 2  
        elif keys[pygame.K_LEFT]:
            angle = np.pi  
        elif keys[pygame.K_RIGHT]:
            angle = 0  
            
        if angle is not None:
            action = (linear_velocity, angle)
    return action

def generate_right_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    x = start_x + radius * np.sin(t)  # NOTE : Virage à droite avec l'axe Y inversé pour pygame
    y = start_y + radius * (1 - np.cos(t)) * strength 
    # y = start_y - radius * (1 - np.cos(t)) * strength  # NOTE : Virage à droite sans l'axe Y inversé
    return x, y

def generate_left_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    x = start_x + radius * np.sin(t)  # NOTE : Virage à gauche avec l'axe Y inversé pour pygame
    y = start_y - radius * (1 - np.cos(t)) * strength 
    # y = start_y + radius * (1 - np.cos(t)) * strength  # NOTE : Virage à gauche sans l'axe Y inversé
    return x, y


def generate_forward(start_x, start_y, length, steps=100):
    t = np.linspace(0, length, steps)
    # x = start_x - t # NOTE : Original forward with pyplot's Y axis
    x = start_x + t
    y = start_y * np.ones(steps)
    return x, y

def generate_backward(start_x, start_y, length, steps=100):
    t = np.linspace(0, length, steps)
    # x = start_x + t # NOTE : Original forward with pyplot's Y axis
    x = start_x - t
    y = start_y * np.ones(steps)
    return x, y

def rotate_points(x, y, rotation_angle):
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle
    return x_rot, y_rot

def generate_one(robot_x, robot_y, how, inertia_angle, shift=False, **kwargs) -> tuple:
    # Default parameter ranges
    radius_min = kwargs.get('radius_min', 2)
    radius_max = kwargs.get('radius_max', 15)
    angle_min = kwargs.get('angle_min', 70)
    angle_max = kwargs.get('angle_max', 110)
    strength_min = kwargs.get('strength_min', 0.5)
    strength_max = kwargs.get('strength_max', 2)
    length_min = kwargs.get('length_min', 3)
    length_max = kwargs.get('length_max', 10)

    # Generate turn parameters
    radius = np.random.uniform(radius_min, radius_max)
    angle = np.random.uniform(angle_min, angle_max)
    strength = np.random.uniform(strength_min, strength_max)
    length = np.random.uniform(length_min, length_max)

    # Generate trajectory based on 'how'
    if 'right' in how:
        x, y = generate_right_turn(start_x=0, start_y=0, radius=radius, angle=angle, strength=strength)
    elif 'left' in how:
        x, y = generate_left_turn(start_x=0, start_y=0, radius=radius, angle=angle, strength=strength)
    elif how == 'forward':
        x, y = generate_forward(start_x=0, start_y=0, length=length)
    elif how == 'backward':
        x, y = generate_backward(start_x=0, start_y=0, length=length)
    else:
        raise ValueError(f"Unknown instruction '{how}'")

    # Rotate turn to align it with the robot's direction
    rotation_angle = inertia_angle
    x_rot, y_rot = rotate_points(x, y, rotation_angle)

    if shift:
        x_rot += robot_x
        y_rot += robot_y

    return x_rot, y_rot, radius, angle

def get_embeddings(model, tokenizer, sentences) -> np.ndarray:
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over the sequence length
    return embeddings.cpu().numpy()