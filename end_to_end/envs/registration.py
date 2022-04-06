from gym.envs.registration import register

# DWA envs
register(
    id="dwa_param_continuous_laser-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousLaser"
)

# Motion control envs
register(
    id="motion_control_continuous_laser-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousLaser"
)