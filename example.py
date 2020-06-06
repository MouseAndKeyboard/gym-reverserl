import gym
import gym_reverserl

env = gym.make("gym_reverserl:mountaincar-v0")
initial_state = env.reset()
print(initial_state)
for step in range(100):
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)