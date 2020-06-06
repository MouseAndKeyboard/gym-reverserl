import gym
import gym_reverserl

# The "optimal" environment policy is totally dependent on the behaviour of the
# agent acting within the environment
def randomAgentPolicy(environment_state, env):
    return env.action_space.sample()

expertEnvPolicy = gym.make("MountainCar-v0") 

# we want to create some buffer of environment_state, agent_action pairs 
# expert env takes the previous state and considers the agent's action to "decide"
# what the next environemnt state should be.
samples = []
for episode in range(100):
    environment_state = expertEnvPolicy.reset()
    for step in range(1000):
        agent_action = randomAgentPolicy(environment_state, expertEnvPolicy)
        samples.append((environment_state, agent_action)) # (previous "action", observation)
        
        # reward and done should be part of the environment_state
        new_environment_state, reward, done, info = expertEnvPolicy.step(agent_action)
        
        
        environment_state = new_environment_state

metaEnv = gym.make("gym_reverserl:mountaincar-v0")
initial_state = metaEnv.reset()
print(initial_state) # initial (environment_state, agent_action) 
for step in range(100):
    environment_state = metaEnv.action_space.sample()
    print(environment_state)
    
    # obs = "agent_action", reward = ??, done = ??
    obs, reward, done, info = metaEnv.step(environment_state)
    
    previous_environment_state, agent_action = obs
    assert environment_state == previous_environment_state # should be the same
    
    print(previous_environment_state, agent_action, reward)