from gym.envs.registration import register

register(
    id='marinenav_env-v0',
    entry_point='marinenav_env.envs:MarineNavEnv2',
)