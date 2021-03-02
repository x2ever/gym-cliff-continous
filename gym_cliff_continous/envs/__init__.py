from gym.envs.registration import register

register(
    id='cliff-continous-v0',
    entry_point='gym_cliff_continous.envs:CliffContinous',
)