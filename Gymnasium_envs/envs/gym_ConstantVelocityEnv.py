import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class ConstantVelocityEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        n_rays: int,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        max_steps: int = 100,
        render_mode: str = None,
    ):
        # Environment constants
        self.MAX_EPISODE_STEPS = max_steps
        self.N_CROWD = n_crowd
        self.INTERCEPTOR_PERCENTAGE = interceptor_percentage
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

        # Observation space
        max_distance = np.linalg.norm([self.WIDTH, self.HEIGHT])
        self.observation_space = spaces.Box(
            low=np.concatenate([
                [0, -np.pi],  # Min agent velocity (speed, velocity angle)
                [0, -np.pi],  # Min goal position (r, theta relative to agent)
                np.zeros(self.N_RAYS * 3)  # Min ray distances
            ]),
            high=np.concatenate([
                [self.MAX_LINEAR_VEL, np.pi],  # Max agent velocity
                [max_distance, np.pi],  # Max goal position
                np.full(self.N_RAYS * 3, max_distance)  # Max ray distances
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
        self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)

        # Goal state
        self.goal_pos = np.random.uniform(
           [-self.W_BORDER + self.PHS, -self.H_BORDER + self.PHS],
           [self.W_BORDER - self.PHS, self.H_BORDER - self.PHS]
        )

        # Observation history
        self.observations = np.zeros((3, self.N_RAYS))

        # Crowd state
        self.crowd_poss = np.zeros((self.N_CROWD, 2))
        collision = True 
        while collision:
            self.crowd_poss = np.random.uniform(
                [-self.W_BORDER, -self.H_BORDER],
                [self.W_BORDER, self.H_BORDER],
                (self.N_CROWD, 2)
            )
            # Check for agent, goal and crowd collisions
            collision = np.any(np.linalg.norm(self.crowd_poss - self.agent_pos, axis=1) < self.PRS * 2) or \
                        np.any(np.linalg.norm(self.crowd_poss - self.goal_pos, axis=1) < self.PRS * 2) or \
                        np.any(np.linalg.norm(self.crowd_poss[:, None] - self.crowd_poss[None, :], axis=-1)[np.triu_indices(self.N_CROWD, k=1)] < self.PHS * 2)
        
        self.crowd_goals = np.random.uniform(
            [-self.W_BORDER, -self.H_BORDER],
            [self.W_BORDER, self.H_BORDER],
            (self.N_CROWD, 2)
        )

        self.crowd_vels = np.random.uniform(
            -self.MAX_LINEAR_VEL,
            self.MAX_LINEAR_VEL,
            self.N_CROWD,
        )

        # Interceptor
        if np.random.rand() < self.INTERCEPTOR_PERCENTAGE:
            interceptor_index = np.random.randint(0, self.N_CROWD)
            direction = self.goal_pos - self.agent_pos
            norm_direction = direction / np.linalg.norm(direction)

            perpendicular_offset = np.random.uniform(-self.PRS, self.PRS)
            perpendicular_vector = np.array([-norm_direction[1], norm_direction[0]])

            interceptor_pos = self.agent_pos + norm_direction * np.linalg.norm(self.goal_pos - self.agent_pos) / 2 + \
                            perpendicular_vector * perpendicular_offset
            # Ensure no collision with the agent
            if np.linalg.norm(interceptor_pos - self.agent_pos) < self.PRS:
                interceptor_pos += perpendicular_vector * self.PRS
            
            self.crowd_poss[interceptor_index] = interceptor_pos


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
        x_crowd_rel, y_crowd_rel = self.crowd_poss[:, 0] - self.agent_pos[0], self.crowd_poss[:, 1] - self.agent_pos[1]
        orthog_dist = np.abs(np.outer(x_crowd_rel, self.RAY_SIN) - np.outer(y_crowd_rel, self.RAY_COS)) # Orthogonal distances from obstacles to rays
        intersections_mask = orthog_dist <= self.PHS # Mask for intersections
        along_dist = np.outer(x_crowd_rel, self.RAY_COS) + np.outer(y_crowd_rel, self.RAY_SIN) # Distance along ray to orthogonal projection
        orthog_to_intersect_dist = np.sqrt(np.maximum(self.PHS**2 - orthog_dist**2, 0)) # Distance from orthogonal projection to intersection
        intersect_distances = np.where(intersections_mask, along_dist - orthog_to_intersect_dist, np.inf) # Distances from ray to intersection if existing
        min_intersect_distances = np.min(np.where(intersect_distances > 0, intersect_distances, np.inf), axis=0) # Minimum distance for each ray to have the closest intersection
        ray_distances = np.minimum(min_intersect_distances, default_distances) # If no intersection, rays collide with border
        self.ray_distances = ray_distances
        
        # Observation history
        self.observations = np.roll(self.observations, shift=-1, axis=0)
        self.observations[-1] = ray_distances

        # Goal relative position in cartesian and then convert to polar
        cart_goal_rel_pos = self.goal_pos - self.agent_pos
        pol_goal_rel_pos = self.c2p(cart_goal_rel_pos)

        return np.concatenate([self.agent_vel, pol_goal_rel_pos, self.observations.flatten()]).astype(np.float32)
            
    def step(self, action):
        # Update agent state
        self.agent_vel = action
        self.agent_pos += self.p2c(self.agent_vel) * self._dt

        # Mise à jour de l'état de la foule
        for i in range(self.N_CROWD):
            direction_to_goal = self.crowd_goals[i] - self.crowd_poss[i]
            norm_direction = direction_to_goal / np.linalg.norm(direction_to_goal)
            self.crowd_poss[i] += norm_direction * self.crowd_vels[i] * self._dt
            
            # Vérifie si un membre de la foule a atteint son objectif
            if np.linalg.norm(self.crowd_poss[i] - self.crowd_goals[i]) < self.PHS:
                # Réaffectation d'un nouvel objectif
                self.crowd_goals[i] = np.random.uniform(
                    [-self.W_BORDER, -self.H_BORDER],
                    [self.W_BORDER, self.H_BORDER],
                    2
                )

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
        # Crowd distance reward
        dist_crowd = np.linalg.norm(self.agent_pos - self.crowd_poss, axis=-1)
        Rc = np.sum(
            (1 - np.exp(self.Cc / dist_crowd)) *\
            (dist_crowd < [self.SCS + self.PHS] * self.N_CROWD)
        )
        # Walls distance reward
        dist_walls = np.array([
            self.W_BORDER - abs(self.agent_pos[0]),
            self.H_BORDER - abs(self.agent_pos[1])
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) * (dist_walls < self.PHS * 2)
        )
        return Rg + Rc + Rw
    
    def _terminate(self):
        # Check for collisions with crowd
        if np.any(np.linalg.norm(self.agent_pos - self.crowd_poss, axis=1) < self.PHS * 2):
            self._is_collided = True

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
        cart_agent_vel = self.p2c(self.agent_vel)
        agent_color = (0, 255, 0)
        agent_center = (
            int((self.agent_pos[0] + self.W_BORDER) * self.RATIO),
            int((self.agent_pos[1] + self.H_BORDER) * self.RATIO)
        )
        agent_radius = self.PHS * self.RATIO
        arrow_pos = (agent_center[0] + int(cart_agent_vel[0] * self.RATIO), 
                agent_center[1] + int(cart_agent_vel[1] * self.RATIO))
        

        # Goal
        goal_color = (0, 0, 255)
        goal_pos_x = int((self.goal_pos[0] + self.W_BORDER) * self.RATIO)
        goal_pos_y = int((self.goal_pos[1] + self.H_BORDER) * self.RATIO)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y - 10), (goal_pos_x + 10, goal_pos_y + 10), 2)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y + 10), (goal_pos_x + 10, goal_pos_y - 10), 2)

        # Crowd
        crowd_color = (255, 0, 0)  # Red
        for pos in self.crowd_poss:
            crowd_center = (
            int((pos[0] + self.W_BORDER) * self.RATIO),
            int((pos[1] + self.H_BORDER) * self.RATIO)
            )
            # Physical space
            crowd_phs = int(self.PHS * self.RATIO)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_phs)

            # Personal space
            crowd_prs = int(self.PRS * self.RATIO)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_prs, 2)

            # Draw dotted circle
            crowd_scs = int(self.SCS * self.RATIO)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_scs, 1)
        
        # Crowd goals
        crowd_goal_color = (255, 100, 0)
        for pos in self.crowd_goals:
            crowd_goal_center = (
            int((pos[0] + self.W_BORDER) * self.RATIO),
            int((pos[1] + self.H_BORDER) * self.RATIO)
            )
            pygame.draw.line(self.window, crowd_goal_color, (crowd_goal_center[0] - 10, crowd_goal_center[1] - 10), (crowd_goal_center[0] + 10, crowd_goal_center[1] + 10), 2)
            pygame.draw.line(self.window, crowd_goal_color, (crowd_goal_center[0] - 10, crowd_goal_center[1] + 10), (crowd_goal_center[0] + 10, crowd_goal_center[1] - 10), 2)

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
        # Dessiner les ailes de la flèche

        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
if __name__ == "__main__":
    env = ConstantVelocityEnv(n_rays=180, n_crowd=4, render_mode="human")
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()
