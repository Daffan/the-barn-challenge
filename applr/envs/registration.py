from gym.envs.registration import register

# DWA envs
register(
    id="dwa_param_continuous_laser-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousLaser"
)

register(
    id="dwa_param_continuous_costmap-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousCostmap"
)

register(
    id="dwa_param_continuous_costmap_resnet-v0",
    entry_point="envs.parameter_tuning_envs:DWAParamContinuousCostmapResnet"
)

# DWA planner assisted motion controller
register(
    id="motion_control_continuous_laser-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousLaser"
)

register(
    id="motion_control_continuous_costmap-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousCostmap"
)

register(
    id="motion_control_continuous_costmap_resnet-v0",
    entry_point="envs.motion_control_envs:MotionControlContinuousCostmapResnet"
)

# Real robot
register(
    id="real_robot_dwa_param_continuous_laser-v0",
    entry_point="envs.real_robot:RealRobotDWAParamContinuousLaser"
)

register(
    id="real_robot_dwa_param_continuous_costmap-v0",
    entry_point="envs.real_robot:RealRobotDWAParamContinuousCostmap"
)

register(
    id="real_robot_motion_control_continuous_laser-v0",
    entry_point="envs.real_robot:RealRobotMotionControlContinuousLaser"
)

register(
    id="real_robot_motion_control_continuous_costmap-v0",
    entry_point="envs.real_robot:RealRobotMotionControlContinuousCostmap"
)
