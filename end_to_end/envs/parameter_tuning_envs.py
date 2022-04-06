import os
from os.path import join
import subprocess

from gym.spaces import Box
import numpy as np
import rospy
import rospkg

from envs.move_base import MoveBase
from envs.jackal_gazebo_envs import JackalGazebo, JackalGazeboLaser

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

class DWAParamContinuous(JackalGazebo):
    def __init__(
        self,
        base_local_planner="base_local_planner/TrajectoryPlannerROS",
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
        self.action_dim = len(param_list)
        super().__init__(**kwargs)
        
        if "init_sim" not in kwargs.keys() or kwargs["init_sim"]:
            self.base_local_planner = base_local_planner
            self.move_base = self.launch_move_base(goal_position=self.goal_position, base_local_planner=self.base_local_planner)
        
        self.param_list = param_list
        self.param_init = param_init

        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )
        
    def launch_move_base(self, goal_position, base_local_planner):
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        launch_file = join(self.BASE_PATH, 'launch', 'move_base_launch.launch')
        self.move_base_process = subprocess.Popen(['roslaunch', launch_file, 'base_local_planner:=' + base_local_planner])
        move_base = MoveBase(goal_position=goal_position, base_local_planner=base_local_planner)
        return move_base

    def kill_move_base(self):
        os.system("pkill -9 move_base")
    
    def _reset_move_base(self):
        # reset the move_base
        # self.kill_move_base()
        # self.move_base = self.launch_move_base(goal_position=self.goal_position, base_local_planner=self.base_local_planner)
        self.move_base.reset_robot_in_odom()
        self._clear_costmap()
        self.move_base.set_global_goal()
        # reset to initial params
        for param_value, param_name in zip(self.param_init, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()

    def _get_info(self):
        info = dict(params=self.params)
        info.update(super()._get_info())
        return info
    
    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.collision_count = 0
        # Reset robot in odom frame clear_costmap
        self.gazebo_sim.reset()
        self.start_time = self.current_time = rospy.get_time()
        pos, psi = self._get_pos_psi()
        
        self.gazebo_sim.unpause()
        self._reset_move_base()
        obs = self._get_observation(pos, psi, np.array(self.param_init))
        self.gazebo_sim.pause()
        
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        self.last_goal_pos = goal_pos
        return obs

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)
        self.gazebo_sim.unpause()
        super()._take_action(action)
        self.gazebo_sim.pause()


class DWAParamContinuousLaser(DWAParamContinuous, JackalGazeboLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
