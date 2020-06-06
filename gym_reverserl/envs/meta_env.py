import gym
from gym import spaces
import numpy as np

class ReverseMountainCar(gym.Env):
        
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        
        self.environemt_observation_space = spaces.Discrete(3)
        self.environemt_action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
    def step(self, environment_action):
        # environemnt_action is an "observation" (like the position & velocity of the cart)
        return environment_action, self.environemt_action_space.sample()
    
    def reset(self):
        initial_environemt_action = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        # return s,a pair
        return initial_environemt_action, self.environemt_observation_space.sample()
        
    def render(self):
        ...