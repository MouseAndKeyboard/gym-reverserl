from gym.envs.registration import register

register(
    id='mountaincar-v0',
    entry_point='gym_reverserl.envs:ReverseMountainCar',
    kwargs={'agent_policy': None}
)
