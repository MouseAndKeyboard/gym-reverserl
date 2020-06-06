import gym
import gym_reverserl

env = gym.make("gym_reverserl:mountaincar-v0")
initial_state = env.reset()
print(initial_state)
for step in range(100):
    action = env.action_space.sample()
    print(action)
    a, b, c, d = env.step(action)
    print(a, b, c ,d)