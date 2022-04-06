from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np
import rospy

from envs.dwa_base_envs import DWABase, DWABaseLaser, DWABaseCostmap, DWABaseCostmapResnet

# A contant dict that define the ranges of parameters
RANGE_DICT = {
    'TrajectoryPlannerROS/max_vel_x': [0.2, 2],
    'TrajectoryPlannerROS/max_vel_theta': [0.314, 3.14],
    'TrajectoryPlannerROS/vx_samples': [4, 12],
    'TrajectoryPlannerROS/vtheta_samples': [8, 40],
    'TrajectoryPlannerROS/path_distance_bias': [0.1, 1.5],
    'TrajectoryPlannerROS/goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.1, 0.6],
    'EBandPlannerROS/max_vel_lin': [0.2, 2],
    'EBandPlannerROS/max_vel_th': [0.314, 3.14],
    "EBandPlannerROS/virtual_mass": [0.2, 1.3],
    "EBandPlannerROS/eband_internal_force_gain": [0.2, 1.8],
    "EBandPlannerROS/eband_external_force_gain": [0.4, 3.6],
    "EBandPlannerROS/costmap_weight": [2, 18]
}

class DWAParamContinuous(DWABase):
    def __init__(
        self, 
        param_init=[0.5, 1.57, 6, 20, 0.75, 1, 0.3],
        param_list=['TrajectoryPlannerROS/max_vel_x', 
                    'TrajectoryPlannerROS/max_vel_theta', 
                    'TrajectoryPlannerROS/vx_samples', 
                    'TrajectoryPlannerROS/vtheta_samples', 
                    'TrajectoryPlannerROS/path_distance_bias', 
                    'TrajectoryPlannerROS/goal_distance_bias', 
                    'inflation_radius'],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.param_init = param_init

        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )

    def _get_info(self):
        info = dict(success=self._get_success(), params=self.params)
        info.update(super()._get_info())
        return info

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters
        self.gazebo_sim.unpause()
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)
        # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()


class DWAParamContinuousLaser(DWAParamContinuous, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DWAParamContinuousCostmap(DWAParamContinuous, DWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class DWAParamContinuousCostmapResnet(DWAParamContinuous, DWABaseCostmapResnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
