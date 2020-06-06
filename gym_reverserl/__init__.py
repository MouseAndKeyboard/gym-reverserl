from gym.env.registration import register

register(
    id='reverse-v0',
    entry_point='gym_reverserl.envs:ReverseMountainCar'
)