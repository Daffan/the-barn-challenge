import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
from os.path import dirname, join, abspath
import subprocess
from gym.spaces import Box, Discrete
from geometry_msgs.msg import Twist

from envs.move_base import MoveBase
from envs.parameter_tuning_envs import RANGE_DICT

class RealRobotEnv(gym.Env):
    def __init__(
        self,
        base_local_planner="base_local_planner/TrajectoryPlannerROS",
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step=1,
        slack_reward=-1,
        failure_reward=-50,
        collision_reward=0,
        success_reward=0,
        verbose=True
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()

        self.base_local_planner = base_local_planner
        self.world_name = world_name
        self.gui = gui
        self.init_position = init_position
        self.goal_position = goal_position
        self.verbose = verbose
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward

        # launch move_base
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        world_name = join(self.BASE_PATH, "worlds", world_name)
        launch_file = join(self.BASE_PATH, 'launch', 'move_base_DWA.launch')

        self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                launch_file,
                                                'base_local_planner:=' + base_local_planner
                                                ])
        time.sleep(10)  # sleep to wait until the gazebo being created

        # initialize the node for gym env
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)

        self.move_base = MoveBase(goal_position=self.goal_position, base_local_planner=base_local_planner)

        # Not implemented
        self.action_space = None
        self.observation_space = None
        self.reward_range = (
            min(slack_reward, failure_reward), 
            success_reward
        )

        self.step_count = 0

    def reset(self):
        """reset the environment
        """
        self.step_count=0
        self.move_base.set_global_goal()
        self._clear_costmap()
        self.start_time = rospy.get_time()
        obs = self._get_observation()
        return obs

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()

    def step(self, action):
        """take an action and step the environment
        """
        self._take_action(action)
        self.step_count += 1
        obs = self._get_observation()
        return obs, 0, False, {}

    def _get_local_goal(self):
        """get local goal in angle
        Returns:
            float: local goal in angle
        """
        local_goal = self.move_base.get_local_goal()
        local_goal = np.array([np.arctan2(local_goal.position.y, local_goal.position.x)])
        return local_goal

    def _get_observation(self):
        raise NotImplementedError()


class RealRobotDWABaseLaser(RealRobotEnv):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip
        
        # 720 laser scan + local goal (in angle)
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(721,),
            dtype=np.float32
        )

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.move_base.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        local_goal = self._get_local_goal()
        
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip # scale to (-0.5, 0.5)
        local_goal = local_goal / (2.0 * np.pi) # scale to (-0.5, 0.5)

        obs = np.concatenate([laser_scan, local_goal])

        return obs


class RealRobotDWABaseCostmap(RealRobotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 720 laser scan + local goal (in angle)
        self.observation_space = Box(
            low=-1,
            high=10,
            shape=(1, 84, 84),
            dtype=np.float32
        )

    def _get_costmap(self):
        PADDING = 62
        costmap = self.move_base.get_costmap().data
        costmap = np.array(costmap, dtype=float).reshape(800, 800)
        # padding to prevent out of index
        occupancy_grid = np.zeros((800 + PADDING * 2, 800 + PADDING * 2), dtype=float)
        occupancy_grid[PADDING:800 + PADDING, PADDING:800 + PADDING] = costmap
        
        global_path = self.move_base.robot_config.global_path
        if len(global_path) > 0:
            path_index = [
                self._to_image_index(*tuple(coordinate), padding=PADDING, size=800)
                for coordinate in global_path
            ]
            for p in path_index:
                occupancy_grid[p[1], p[0]] = -100

        X, Y = self.move_base.robot_config.X, self.move_base.robot_config.Y  # robot position
        X, Y = self._to_image_index(X, Y, padding=PADDING, size=800)
        occupancy_grid = occupancy_grid[
            Y - PADDING:Y + PADDING,
            X - PADDING:X + PADDING
        ]
        obstacles_index = np.where(occupancy_grid == 100)
        path_index = np.where(occupancy_grid == -100)
        occupancy_grid[:, :] = 1
        occupancy_grid[obstacles_index] = 10
        occupancy_grid[path_index] = 0
        psi = self.move_base.robot_config.PSI
        occupancy_grid = self.rotate_image(occupancy_grid, psi/np.pi*180)
        w, h = occupancy_grid.shape[0], occupancy_grid.shape[1]
        occupancy_grid = occupancy_grid[w//2-42:w//2+42, h//2-42:h//2+42]
        occupancy_grid = occupancy_grid.reshape(84, 84)
        assert occupancy_grid.shape == (84, 84), "x, y, z: %d, %d, %d; X, Y: %d, %d" %(occupancy_grid.shape[0], occupancy_grid.shape[1], occupancy_grid.shape[2], X, Y)
        
        recolor_map = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
        )
        return recolor_map[(occupancy_grid).astype(int)]
   
    def rotate_image(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def _to_image_index(self, x, y, padding=42, size=800):
        X, Y = int(x * size // 40) + size // 2 + padding, int(y * size // 40) + size // 2 + padding
        X, Y = min(size - 1 + padding, X), min(size - 1 + padding, Y)
        X, Y = max(padding, X), max(padding, Y)
        return X, Y

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        costmap = self._get_costmap()
        # for now we skip the local goal temperally
        # local_goal = local_goal / (2.0 * np.pi) # scale to (-0.5, 0.5)
        obs = costmap
        return obs

    def visual_costmap(self, costmap):
        from matplotlib import pyplot as plt
        import cv2

        costmap = (costmap * 100).astype(int).reshape(-1, 84, 84)
        costmap = np.transpose(costmap, axes=(1, 2, 0)) + 100
        costmap = np.repeat(costmap, 3, axis=2)
        plt.imshow(costmap, origin="bottomleft")
        plt.show(block=False)
        plt.pause(.5)


class RealRobotMotionControlContinuous(RealRobotEnv):
    def __init__(self, collision_reward=-0.1, **kwargs):
        super().__init__(**kwargs)
        self.collision_reward = collision_reward
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.params = None
        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([-0.2, -3.14]),
            high=np.array([2, 3.14]),
            dtype=np.float32
        )

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.move_base.make_plan()
        self._clear_costmap()
        self.start_time = rospy.get_time()
        obs = self._get_observation()
        return obs

    def _take_action(self, action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._cmd_vel_pub.publish(cmd_vel_value)
        self.move_base.make_plan()
        rospy.sleep(self.time_step)


class RealRobotMotionControlContinuousLaser(RealRobotMotionControlContinuous, RealRobotDWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotMotionControlContinuousCostmap(RealRobotMotionControlContinuous, RealRobotDWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotDWAParamContinuous(RealRobotEnv):
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

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)
        # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)


class RealRobotDWAParamContinuousLaser(RealRobotDWAParamContinuous, RealRobotDWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotDWAParamContinuousCostmap(RealRobotDWAParamContinuous, RealRobotDWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)