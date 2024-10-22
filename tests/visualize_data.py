import numpy as np
from matplotlib import pyplot as plt

from utils import generate_one

# N Robots visualisation
num_robots = 100
robot_spawn = 0
bounds = 20
robot_positions = np.random.uniform(-robot_spawn, robot_spawn, (num_robots, 2))
inertia_angles = np.random.uniform(-np.pi, np.pi, num_robots)
inertia_length = 2
scatter = False

plt.figure(figsize=(10, 10))

for i in range(num_robots):
    # Robot position
    robot_x, robot_y = robot_positions[i]
    inertia_angle = inertia_angles[i]
    
    x_rot, y_rot,radius, angle = generate_one(robot_x, robot_y, 'right', inertia_angle, shift=True, radius_min=5, radius_max=10, angle_min=360, angle_max=360, strength_min=0.5, strength_max=2)
    # If the turn is out of bounds
    while np.any(np.abs(x_rot) > bounds * 0.9) or np.any(np.abs(y_rot) > bounds * 0.9):
        x_rot, y_rot, radius, angle = generate_one(robot_x, robot_y, 'right', inertia_angle, shift=True)

    if scatter:
        indices = np.linspace(0, len(x_rot) - 1, 5, dtype=int)
        x_rot = x_rot[indices]
        y_rot = y_rot[indices]
        plt.scatter(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')
    else:
        plt.plot(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')

    plt.plot(robot_x, robot_y, 'go', markersize=10)
    plt.arrow(robot_x, robot_y, inertia_length * np.cos(inertia_angle), inertia_length * np.sin(inertia_angle),
              head_width=0.5, head_length=0.5, fc='blue', ec='blue')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
# Draw a red square around the bounds
plt.plot([-bounds, bounds, bounds, -bounds, -bounds], [-bounds, -bounds, bounds, bounds, -bounds], 'r')
plt.axis([-bounds - 5, bounds + 5, -bounds - 5, bounds + 5])
# Invert Y axis to match pygame's coordinate system
plt.gca().invert_yaxis()
plt.show()
