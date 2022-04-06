from gym.spaces import Box
import numpy as np
import os
import subprocess
import time

import rospy
import rospkg
from geometry_msgs.msg import Twist, PoseStamped, Quaternion

from envs.jackal_gazebo_envs import JackalGazebo, JackalGazeboLaser
from envs.move_base import MoveBase
from envs.gazebo_simulation import GazeboSimulation

class MotionControlContinuous(JackalGazebo):
    def __init__(self, min_v=-1, max_v=2, min_w=-3.14, max_w=3.14, **kwargs):
        self.action_dim = 2
        super().__init__(**kwargs)
        rospy.init_node('e2e', anonymous=True) #, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.range_dict = RANGE_DICT = {
            "linear_velocity": [min_v, max_v],
            "angular_velocity": [min_w, max_w],
        }
        self.action_space = Box(
            low=np.array([RANGE_DICT["linear_velocity"][0], RANGE_DICT["angular_velocity"][0]]),
            high=np.array([RANGE_DICT["linear_velocity"][1], RANGE_DICT["angular_velocity"][1]]),
            dtype=np.float32
        )

        # if "init_sim" not in kwargs.keys() or kwargs["init_sim"]:
        self.base_local_planner = "base_local_planner/TrajectoryPlannerROS"
        self.move_base = self.launch_move_base(goal_position=self.goal_position, base_local_planner=self.base_local_planner)
        time.sleep(5)
        self.gazebo_sim = GazeboSimulation()

    def launch_move_base(self, goal_position, base_local_planner):
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        launch_file = os.path.join(self.BASE_PATH, 'launch', 'move_base_DWA.launch')
        self.move_base_process = subprocess.Popen(['roslaunch', launch_file, 'base_local_planner:=' + base_local_planner])
        move_base = MoveBase(goal_position=goal_position, base_local_planner=base_local_planner)
        return move_base

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.collision_count = 0
        # Reset robot in odom frame clear_costmap
        # self.gazebo_sim.reset()
        self.start_time = self.current_time = rospy.get_time()
        pos, psi = self._get_pos_psi()
        
        # self.gazebo_sim.unpause()
        self.move_base.reset_robot_in_odom()
        self.move_base.make_plan()
        self._clear_costmap()
        obs = self._get_observation(0, 0, np.array([0, 0]))
        # self.gazebo_sim.pause()
        
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        self.last_goal_pos = goal_pos
        return obs

    def _get_observation(self, pos, psi, action):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip * 2 # scale to (-1, 1)
        
        # goal_pos = self.transform_goal(self.world_frame_goal, pos, psi) / 5.0 - 1  # roughly (-1, 1) range
        goal_pos = self.move_base.get_global_path()[-1] / 5.0 - 1
        
        bias = (self.action_space.high + self.action_space.low) / 2.
        scale = (self.action_space.high - self.action_space.low) / 2.
        action = (action - bias) / scale
        
        obs = [laser_scan, goal_pos, action]
        
        obs = np.concatenate(obs)

        return obs

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()

    def _take_action(self, action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        # self.gazebo_sim.unpause()
        self._cmd_vel_pub.publish(cmd_vel_value)
        super()._take_action(action)  # this will wait util next time step
        self.move_base.make_plan()
        # self.gazebo_sim.pause()


class MotionControlContinuousLaser(MotionControlContinuous, JackalGazeboLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
