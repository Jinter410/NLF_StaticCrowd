from gymnasium.envs.registration import register

register(
    id="StaticCrowd-v0",
    entry_point="Gymnasium_envs.envs:StaticCrowdEnv"
)

register(
    id="ConstantVelocity-v0",
    entry_point="Gymnasium_envs.envs:ConstantVelocityEnv"
)

register(
    id="Navigation-v0",
    entry_point="Gymnasium_envs.envs:NavigationEnv"
)

