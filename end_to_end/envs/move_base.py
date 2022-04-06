import rospy
import actionlib
import numpy as np
import scipy.signal

import dynamic_reconfigure.client
from robot_localization.srv import SetPose

from std_srvs.srv import Empty
from geometry_msgs.msg import Quaternion, Pose, PoseWithCovarianceStamped, Twist, PoseStamped
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from nav_msgs.srv import GetPlan

def _create_MoveBaseGoal(x, y, angle):
    """
    Create a MoveBaseGoal with x, y position and yaw rotation (in degrees).
    Returns a MoveBaseGoal
    """
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = 'odom' # Note: the frame_id must be map
    mb_goal.target_pose.pose.position.x = x
    mb_goal.target_pose.pose.position.y = y
    mb_goal.target_pose.pose.position.z = 0 # z must be 0.0 (no height in the map)

    mb_goal.target_pose.pose.orientation = Quaternion(0, 0, np.sin(angle/2.), np.cos(angle/2.))

    return mb_goal

def _create_PoseWithCovarianceStamped():
    """
    Create initial pose in odometery frame (used to reset)
    """
    a = PoseWithCovarianceStamped()
    a.header.frame_id = 'odom'
    a.pose.pose.position.x = 0.0
    a.pose.pose.position.y = 0.0
    a.pose.pose.position.z = 0.0
    a.pose.pose.orientation.x = 0.0
    a.pose.pose.orientation.y = 0.0
    a.pose.pose.orientation.z = 0.0
    a.pose.pose.orientation.w = 0.0
    return a

class Robot_config():
    """This is a class that tracks the jackal robot status
    """
    def __init__(self):
        self.X = 0 # inertia frame
        self.Y = 0
        self.Z = 0
        self.PSI = 0
        self.global_path = []
        self.gx = 0 # body frame
        self.gy = 0
        self.gp = 0
        self.los = 1
        self.bad_vel = 0
        self.vel_counter = 1  # avoid divided by zero error
        self.collision_count = 0
        self.qt = (0, 0, 0, 0)
        self.last_lg = (self.X, self.Y)

    def get_robot_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.Z = msg.pose.pose.position.z
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        self.qt = (q1, q2, q3, q0)

    def get_global_path(self, msg):
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
        gp = np.array(gp)
        x = gp[:,0]
        try:
            xhat = scipy.signal.savgol_filter(x, 19, 3)
        except:
            xhat = x
        y = gp[:,1]
        try:
            yhat = scipy.signal.savgol_filter(y, 19, 3)
        except:
            yhat = y
        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        self.global_path = gphat

def transform_lg(wp, X, Y, PSI):
    R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
    R_i2r = np.linalg.inv(R_r2i)
    pi = np.matrix([[wp[0]], [wp[1]], [1]])
    pr = np.matmul(R_i2r, pi)
    lg = np.array([pr[0,0], pr[1, 0]])
    return lg

def transform_gp(gp, X, Y, PSI):
    R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
    R_i2r = np.linalg.inv(R_r2i)

    pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
    pr = np.matmul(R_i2r, pi.T)
    return np.asarray(pr[:2, :])


class MoveBase():

    def __init__(self, goal_position = [6, 6, 0], base_local_planner="base_local_planner/TrajectoryPlannerROS"):
        self.goal_position = goal_position
        self.base_local_planner = base_local_planner.split("/")[-1]
        self.planner_client = dynamic_reconfigure.client.Client('/move_base/' + self.base_local_planner)
        self.local_costmap_client = dynamic_reconfigure.client.Client('move_base/local_costmap/inflater_layer')
        self.global_costmap_client = dynamic_reconfigure.client.Client('move_base/global_costmap/inflater_layer')
        self.nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.global_goal = _create_MoveBaseGoal(goal_position[0], goal_position[1], goal_position[2])
        self._reset_odom = rospy.ServiceProxy('/set_pose', SetPose)
        self._clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        self.robot_config = Robot_config()
        self.sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, self.robot_config.get_robot_status)
        # self.sub_gp = rospy.Subscriber("/move_base/" + self.base_local_planner + "/global_plan", Path, self.robot_config.get_global_path)
        self.sub_gp = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.robot_config.get_global_path)
        self._make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

    def make_plan(self):
        # get_plan = GetPlan()

        start = PoseStamped()
        start.header.frame_id = "odom"
        start.pose.position.x = self.robot_config.X
        start.pose.position.y = self.robot_config.Y
        start.pose.position.z = self.robot_config.Z
        x, y, z, w = self.robot_config.qt
        start.pose.orientation.x = x
        start.pose.orientation.y = y
        start.pose.orientation.z = z
        start.pose.orientation.w = w

        goal = PoseStamped()
        x, y, angle = self.goal_position
        goal.header.frame_id = "odom"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0
        goal.pose.orientation = Quaternion(0, 0, np.sin(angle/2.), np.cos(angle/2.))
        tolerance = 0.5
        
        rospy.wait_for_service('/move_base/make_plan')
        try:
            self._make_plan(start, goal, tolerance)
        except rospy.ServiceException:
            print ("/make_plan service call failed")

    def set_navi_param(self, param_name, param):

        if param_name != 'inflation_radius':
            self.planner_client.update_configuration({param_name.split("/")[-1]: param})
            rospy.set_param('/move_base/' + param_name, param)

            if param_name == 'max_vel_theta':
                self.planner_client.update_configuration({'min_vel_theta': -param})
                rospy.set_param('/move_base/' + 'min_vel_theta', -param)
        else:
            self.global_costmap_client.update_configuration({param_name: param})
            self.local_costmap_client.update_configuration({param_name: param})
            rospy.set_param('/move_base/global_costmap/inflater_layer/' + param_name, param)
            rospy.set_param('/move_base/local_costmap/inflater_layer/' + param_name, param)

    def get_navi_param(self, param_name):
        if param_name != 'inflation_radius':
            param = rospy.get_param('/move_base/' + param_name)
        else:
            param = rospy.get_param('/move_base/global_costmap/inflater_layer/' + param_name)
        return param

    def set_global_goal(self):
        self.nav_as.wait_for_server()
        try:
            self.nav_as.send_goal(self.global_goal)
        except (rospy.ServiceException) as e:
            print ("/move_base service call failed")

    def reset_robot_in_odom(self):
        rospy.wait_for_service('/set_pose')
        try:
            self._reset_odom(_create_PoseWithCovarianceStamped())
        except rospy.ServiceException:
            print ("/set_pose service call failed")
        self.robot_config.X = 0
        self.robot_config.Y = 0
        self.robot_config.Z = 0
        # clear vel count history
        self.robot_config.bad_vel = 0
        self.robot_config.vel_counter = 0

    def clear_costmap(self):
        rospy.wait_for_service('/move_base/clear_costmaps')
        try:
            self._clear_costmap()
        except rospy.ServiceException:
            print ("/clear_costmaps service call failed")

    def reset_global_goal(self, goal_position = [6, 6, 0]):
        self.global_goal = _create_MoveBaseGoal(goal_position[0], goal_position[1], goal_position[2])

    def get_local_goal(self):
        """Get the local goal coordinate relative to the robot's current location

        Returns:
            [Pose msg]: pose msg with attributes x, y, and orientaton
        """
        gp = self.robot_config.global_path
        X = self.robot_config.X
        Y = self.robot_config.Y
        PSI = self.robot_config.PSI
        los = self.robot_config.los
        
        lg_x, lg_y = self.robot_config.last_lg
        dist_last_lg = (X - lg_x) ** 2 + (Y - lg_y) ** 2

        lg_x = 0
        lg_y = 0
        if len(gp)>0:
            lg_flag = 0
            for wp in gp:
                dist = (np.array(wp)-np.array([X, Y]))**2
                dist = np.sum(dist, axis=0)
                dist = np.sqrt(dist)
                if dist > los:
                    lg_flag = 1
                    self.robot_config.last_lg = wp
                    lg = transform_lg(wp, X, Y, PSI)
                    lg_x = lg[0]
                    lg_y = lg[1]
                    break
            if lg_flag == 0:
                self.robot_config.last_lg = gp[-1]
                lg = transform_lg(gp[-1], X, Y, PSI)
                lg_x = lg[0]
                lg_y = lg[1]

        local_goal = Pose()
        local_goal.position.x = lg_x
        local_goal.position.y = lg_y
        local_goal.orientation.w = 1
        return local_goal, dist_last_lg

    def get_global_path(self):
        gp = self.robot_config.global_path
        gp = transform_gp(gp, self.robot_config.X, self.robot_config.Y, self.robot_config.PSI)
        return gp.T

    def get_costmap(self):
        cm = None
        while cm is None:
            try:
                cm = rospy.wait_for_message("/move_base/global_costmap/costmap", OccupancyGrid, timeout=5)
            except: 
                pass
        return cm
