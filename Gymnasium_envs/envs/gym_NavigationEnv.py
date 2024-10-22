import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        n_rays: int,
        width: int = 20,
        height: int = 20,
        max_steps: int = 100,
        render_mode: str = None,
    ):
        # Environment constants
        self.MAX_EPISODE_STEPS = max_steps
        self.N_RAYS = n_rays
        self.RAY_ANGLES = np.linspace(0, 2 * np.pi, n_rays, endpoint=False) + 1e-6 # To avoid div by zero
        self.RAY_COS = np.cos(self.RAY_ANGLES)
        self.RAY_SIN = np.sin(self.RAY_ANGLES)
        self.WIDTH = width
        self.HEIGHT = height
        self.W_BORDER = self.WIDTH / 2
        self.H_BORDER = self.HEIGHT / 2
        self.PHS = 0.4
        self.PRS = 1.4
        self.SCS = 1.9
        # Physics
        self._dt = 0.1
        self.MAX_LINEAR_VEL = 3.0 # m/s
        self.MAX_ANGULAR_VEL = 1.5 # rad/s
        self.MAX_GOAL_VEL = 0.5 # m/s
        # Reward constants
        self.COLLISION_REWARD = -10
        self.Cc = 2 * self.PHS * \
            np.log(-self.COLLISION_REWARD / self.MAX_EPISODE_STEPS + 1)
        self.Cg = -(1 - np.exp(self.Cc / self.SCS)) /\
            np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2)
        self.TASK_COMPLETION_REWARD = -self.COLLISION_REWARD
        # Action space (linear and angular velocity)
        self.action_space = spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([self.MAX_LINEAR_VEL, np.pi]),
            dtype=np.float32
        )
        self.coordinate_list = None

        # Observation space
        max_distance = np.linalg.norm([self.WIDTH, self.HEIGHT])
        self.observation_space = spaces.Box(
            low=np.concatenate([
                [0, -np.pi],  # Min agent velocity (speed, velocity angle)
                [0, -np.pi],  # Min goal position (r, theta relative to agent)
                np.zeros(self.N_RAYS)  # Min ray distances
            ]),
            high=np.concatenate([
                [self.MAX_LINEAR_VEL, np.pi],  # Max agent velocity
                [max_distance, np.pi],  # Max goal position
                np.full(self.N_RAYS, max_distance)  # Max ray distances
            ]),
            dtype=np.float32
        )
        # Plotting
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.RATIO = 30

    def c2p(self, cart):
        r = np.linalg.norm(cart)
        theta = np.arctan2(cart[1], cart[0])
        return np.array([r, theta])

    def p2c(self, pol):
        x = pol[0] * np.cos(pol[1])
        y = pol[0] * np.sin(pol[1])
        return np.array([x, y])

    def reset(self, seed=None, options=None):
        # Seeding
        super().reset(seed=seed)

        # Episode variables
        self._steps = 0
        self._reward = 0
        self._total_reward = 0
        self._goal_reached = False
        self._is_collided = False

        # Agent state
        # self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)
        self.agent_pos = self.goal_pos = np.random.uniform(
           [-self.W_BORDER + self.PHS, -self.H_BORDER + self.PHS],
           [self.W_BORDER - self.PHS, self.H_BORDER - self.PHS]
        )

        # Goal state
        self.goal_pos = np.random.uniform(
           [-self.W_BORDER + self.PHS, -self.H_BORDER + self.PHS],
           [self.W_BORDER - self.PHS, self.H_BORDER - self.PHS]
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _get_obs(self):
        # Vectorized ray distances
        default_distances = np.min([
            (self.W_BORDER - np.where(self.RAY_COS > 0, self.agent_pos[0], -self.agent_pos[0])) / np.abs(self.RAY_COS),
            (self.H_BORDER - np.where(self.RAY_SIN > 0, self.agent_pos[1], -self.agent_pos[1])) / np.abs(self.RAY_SIN)
        ], axis=0)
        ray_distances = default_distances # Rays collide with border
        self.ray_distances = ray_distances
        
        # Goal relative position in cartesian and then convert to polar
        cart_goal_rel_pos = self.goal_pos - self.agent_pos
        pol_goal_rel_pos = self.c2p(cart_goal_rel_pos)

        return np.concatenate([self.agent_vel, pol_goal_rel_pos, ray_distances]).astype(np.float32)
            
    def step(self, action):
        self.agent_vel = action
        self.agent_pos += self.p2c(self.agent_vel) * self._dt

        terminated = self._terminate()
        reward = self._get_reward()
        info = self._get_info()
        observation = self._get_obs()
        self._steps += 1
        truncated = self._steps >= self.MAX_EPISODE_STEPS
        self._reward = reward
        self._total_reward += reward

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    
    def _get_reward(self):
        if self._goal_reached:
            return self.TASK_COMPLETION_REWARD
        if self._is_collided:
            return self.COLLISION_REWARD
        
        dg = np.linalg.norm(self.agent_pos - self.goal_pos)
        # Goal distance reward
        Rg = -self.Cg * dg ** 2
        # Walls distance reward
        dist_walls = np.array([
            self.W_BORDER - abs(self.agent_pos[0]),
            self.H_BORDER - abs(self.agent_pos[1])
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) * (dist_walls < self.PHS * 2)
        )
        return Rg + Rw
    
    def _terminate(self):
        # Check for collisions with walls
        if np.any(np.abs(self.agent_pos) > np.array([self.W_BORDER, self.H_BORDER]) - self.PHS):
            self._is_collided = True

        # Check for goal reached
        if (np.linalg.norm(self.agent_pos - self.goal_pos) < self.PHS) and \
            (np.linalg.norm(self.agent_vel) < self.MAX_GOAL_VEL):
            self._goal_reached = True
        return self._goal_reached or self._is_collided
    
    def _get_info(self):
        return {
            "goal_reached": self._goal_reached, 
            "collision": self._is_collided, 
            "steps": self._steps, 
            "dist_to_goal": np.linalg.norm(self.agent_pos - self.goal_pos)
        }
    def set_coordinate_list(self, coordinates):
        """Set a list of coordinates to be plotted during rendering."""
        self.coordinate_list = coordinates
    
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.WIDTH * self.RATIO, self.HEIGHT * self.RATIO))
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        pygame.display.set_caption(f"Steps: {self._steps} Reward: {self._reward:.5f} Total Reward: {self._total_reward:.3f}")
        
        self.window.fill((245,245,245))

        # Agent
        agent_color = (0, 255, 0)
        agent_center = (
            int((self.agent_pos[0] + self.W_BORDER) * self.RATIO),
            int((self.agent_pos[1] + self.H_BORDER) * self.RATIO)
        )
        agent_radius = self.PHS * self.RATIO
        agent_speed, agent_angle = self.agent_vel
        arrow_pos = (
            agent_center[0] + int(agent_speed * np.cos(agent_angle) * self.RATIO),  
            agent_center[1] + int(agent_speed * np.sin(agent_angle) * self.RATIO)
        )
        
        # Goal
        goal_color = (0, 0, 255)
        goal_pos_x = int((self.goal_pos[0] + self.W_BORDER) * self.RATIO)
        goal_pos_y = int((self.goal_pos[1] + self.H_BORDER) * self.RATIO)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y - 10), (goal_pos_x + 10, goal_pos_y + 10), 2)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y + 10), (goal_pos_x + 10, goal_pos_y - 10), 2)
        
        # Wall borders
        wall_color = (0, 0, 0)
        pygame.draw.rect(self.window, wall_color, (self.PHS * self.RATIO, self.PHS * self.RATIO, (self.WIDTH - 2 * self.PHS) * self.RATIO, (self.HEIGHT - 2 * self.PHS) * self.RATIO), 1)

        # Rays
        ray_color = (128, 128, 128)  # Gray
        for angle, distance in zip(self.RAY_ANGLES, self.ray_distances):
            end_x = agent_center[0] + distance * self.RATIO * np.cos(angle)
            end_y = agent_center[1] + distance * self.RATIO * np.sin(angle)
            pygame.draw.line(self.window, ray_color, agent_center, (int(end_x), int(end_y)), 1)

        # Draw agent above rays
        pygame.draw.circle(self.window, agent_color, agent_center, agent_radius)
        pygame.draw.line(self.window, (0, 150, 0), agent_center, arrow_pos, 3)
        
        predefined_colors = [
            (255, 165, 0),   # Orange
            (197, 197, 8),   # Jaune
            (255, 0, 0),     # Rouge
            (8, 120, 224),   # Bleu
            (85, 238, 243),  # Bleu ciel
            (133, 11, 204),  # Violet
            (0, 0, 0),       # Noir
            (128, 128, 128)  # Gris
        ]

        # Afficher les points s'ils existent
        if self.coordinate_list is not None:
            if isinstance(self.coordinate_list, list):
                point_color = (64, 0, 128)
                for coord in self.coordinate_list:
                    point_x = int((coord[0] + self.W_BORDER) * self.RATIO)
                    point_y = int((coord[1] + self.H_BORDER) * self.RATIO)
                    pygame.draw.circle(self.window, point_color, (point_x, point_y), 5)

            elif isinstance(self.coordinate_list, dict):
                labels = list(self.coordinate_list.keys())
                # Créer une correspondance entre les labels et les couleurs prédéfinies
                label_color_mapping = {label: predefined_colors[i % len(predefined_colors)] for i, label in enumerate(labels)}

                # Cadre pour afficher la légende
                legend_rect = pygame.Rect(5, 5, 220, 18 + 18 * len(self.coordinate_list))
                pygame.draw.rect(self.window, (233, 233, 233), legend_rect)
                
                # Affichage de la légende
                font = pygame.font.SysFont(None, 24)
                for i, (label, color) in enumerate(label_color_mapping.items()):
                    label_surface = font.render(label, True, (0, 0, 0))
                    pygame.draw.circle(self.window, color, (20, 20 + i * 20), 5)
                    self.window.blit(label_surface, (30, 10 + i * 20))

                # Affichage des points pour chaque trajectoire
                for label, coordinates in self.coordinate_list.items():
                    point_color = label_color_mapping[label]
                    for coord in coordinates:
                        point_x = int((coord[0] + self.W_BORDER) * self.RATIO)
                        point_y = int((coord[1] + self.H_BORDER) * self.RATIO)
                        pygame.draw.circle(self.window, point_color, (point_x, point_y), 5)

        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
if __name__ == "__main__":
    env = NavigationEnv(n_rays=180, n_crowd=4, render_mode="human")
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()
