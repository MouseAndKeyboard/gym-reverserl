import gym
import gym_reverserl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)


    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def clean(x):
    position = x[0]
    velocity = x[1]
    action = x[2]

    x[0] = (x[0] + 0.3) * 1.11
    x[1] = x[1] * 14.29
    return [position, velocity, *np.eye(3)[x[2]]]


def train(net, data, epochs=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(data):
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, torch.Tensor(labels))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 9:
                print(f'[{epoch+1}, {i+1}] {total_loss/200}')
                total_loss = 0

    print('done')
    torch.save(net.state_dict(), './mountain_car.pth')
   
# The "optimal" environment policy is totally dependent on the behaviour of the
# agent acting within the environment
def randomAgentPolicy(environment_state, env):
    return env.action_space.sample()

expertEnvPolicy = gym.make("MountainCar-v0")

# we want to create some buffer of environment_state, agent_action pairs 
# expert env takes the previous state and considers the agent's action to "decide"
# what the next environemnt state should be.
train_set = []
for episode in range(200):
    previous_environment_state = expertEnvPolicy.reset()
    minibatch_x = []
    minibatch_labels = []
    for step in range(400):
        agent_action = randomAgentPolicy(previous_environment_state, expertEnvPolicy)
        
        # reward and done could possibly be part of the environment_state
        environment_state, reward, done, info = expertEnvPolicy.step(agent_action)

        minibatch_x.append(clean([*previous_environment_state, agent_action]))
        minibatch_labels.append(environment_state)

        previous_environment_state = environment_state
    train_set.append([minibatch_x, minibatch_labels])


# Now we can do "behavioural cloning" on the expert. This will amount to predicting the next state given a previous state+action
bc = Net()
bc = train(bc, train_set)


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
    # print(previous_environment_state, agent_action, environment_state)
    
    obs, reward, done, info = metaEnv.step(environment_state)
    previous_environment_state, agent_action = obs
    assert (environment_state == previous_environment_state).all() # should be the same
