from gym.env.registration import register

register(
    id='mountaincar-v0',
    entry_point='gym_reverserl.envs:ReverseMountainCar'
)