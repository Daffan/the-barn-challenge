import numpy as np
import time
# import scipy

import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist


class StepRecorder:
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.Z = 0
        self.PSI = 0

        self.laser_scan = None
        self.global_path = None

    def laser_scan_callback(self, msg: LaserScan):
        self.laser_scan = msg.ranges

    def robot_stat_callback(self, msg: Odometry):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.Z = msg.pose.pose.position.z
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))

    def transform_lg(self, wp, X, Y, PSI):
        R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[wp[0]], [wp[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0, 0], pr[1, 0]])
        return lg

    def global_path_callback(self, msg: Path):
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
        if len(gp) == 0:
            return -1
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

    def get_obs(self):
        if self.laser_scan is None:
            print("laser is None")
            return None
        if self.global_path is None:
            print("path is None")
            return None
        laser_scan = self.laser_scan

        goal = self.global_path[-1]  # Goal is the last point on the global path
        # transform the goal coordinates in robot's frame
        goal = self.transform_lg(goal, self.X, self.Y, self.PSI).reshape(-1)

        # observation is laser_scan + goal coordinate
        return np.concatenate([laser_scan, goal])

if __name__ == "__main__":
    FREQUENCY = 5.0  # In Hz

    rospy.init_node('step_recording', anonymous=True)
    rospy.set_param('/use_sim_time', True)

    step_recorder = StepRecorder()

    robot_state_sub = rospy.Subscriber(
        "/odometry/filtered",
        Odometry,
        step_recorder.robot_stat_callback,
        queue_size=1
    )
    laser_scan_sub = rospy.Subscriber(
        "/front/scan",
        LaserScan,
        step_recorder.laser_scan_callback,
        queue_size=1
    )
    global_path_sub = rospy.Subscriber(
        "/move_base/NavfnROS/plan",
        Path,
        step_recorder.global_path_callback,
        queue_size=1
    )
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    ## Initialize your policy here

    while step_recorder.get_obs() is None:
        time.sleep(0.1)

    time = rospy.get_time()
    while not rospy.is_shutdown():
        obs = step_recorder.get_obs()
        v, w = 0.5, 0  # sample action from your policy
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = v
        cmd_vel_value.angular.z = w
        cmd_vel_pub.publish(cmd_vel_value)

        while rospy.get_time() - time < 1/FREQUENCY:
            rospy.sleep(0.1/FREQUENCY)

        time = rospy.get_time()
