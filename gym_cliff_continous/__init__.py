from gym.envs.registration import register

register(
    id='cliff-continuous-v0',
    entry_point='gym_cliff_continous.envs:CliffContinuous',
)