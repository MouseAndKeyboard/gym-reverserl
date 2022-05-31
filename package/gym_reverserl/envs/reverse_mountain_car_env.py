import gym
from gym import spaces
import numpy as np

class ReverseMountainCar(gym.Env):
        
    def __init__(self, agent_policy):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)


        self.agent_action_space = spaces.Discrete(3)

        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Tuple((self.action_space, self.agent_action_space))



        if agent_policy == None:
            raise ValueError('Agent Policy required as parameter. Policy : observation -> action.')
        self.agent_policy = agent_policy

    def step(self, environment_action):
        # environemnt_action is an "observation" (like the position & velocity of the cart)
        
        # run the agent's policy here:
        agent_action = self.agent_policy(environment_action, self)

        observation = (environment_action, agent_action)

        assert self.observation_space.contains(observation)

        return observation, 0, False, {}
    
    def reset(self):
        initial_environment_action = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        initial_agent_action = self.agent_policy(initial_environment_action, self)

        initial_observation = (initial_environment_action, initial_agent_action)

        assert self.observation_space.contains(initial_observation)

        # return s,a pair
        return initial_observation
        
    def render(self):
        ...
