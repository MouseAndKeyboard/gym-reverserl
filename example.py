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
expert_observations = []
expert_actions = []
for episode in range(1):
    previous_environment_state = expertEnvPolicy.reset()
    for step in range(1):
        agent_action = randomAgentPolicy(previous_environment_state, expertEnvPolicy)
        
        
        # reward and done could possibly be part of the environment_state
        environment_state, reward, done, info = expertEnvPolicy.step(agent_action)

        expert_observations.append((previous_environment_state), agent_action)
        expert_actions.append(environment_state)
       
        previous_environment_state = environment_state



# Now we can do "behavioural cloning" on the expert. This will amount to predicting the next state given a previous state+action


def randomAgentPolicy(environment_state, env):
    return env.agent_action_space.sample()

metaEnv = gym.make("gym_reverserl:mountaincar-v0", agent_policy=randomAgentPolicy)
def customNextEnvStatePolicy(previous_env_state, agent_action):
    return metaEnv.action_space.sample()

obs = metaEnv.reset()
previous_environment_state, agent_action = obs
for step in range(100):
    # select next environment_state based on obs
    environment_state = customNextEnvStatePolicy(previous_environment_state, agent_action)
    print(previous_environment_state, agent_action, environment_state)
    
    obs, reward, done, info = metaEnv.step(environment_state)
    previous_environment_state, agent_action = obs
    assert (environment_state == previous_environment_state).all() # should be the same
