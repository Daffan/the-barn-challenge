import time
import argparse
import subprocess
from os.path import join

import rospy
import rospkg

from gazebo_simulation import GazeboSimulation

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [0, 10]  # relative to the initial position

def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'test BARN navigation challenge')
    parser.add_argument('--navigation_stack', type=str, default="jackal_helper/launch/move_base_DWA.launch",\
        help="path to the launch file of the tested navigation stack (relative to the path of this repo)")
    parser.add_argument('--world_idx', type=int, default=0)
    parser.add_argument('--gui', action="store_true")
    args = parser.parse_args()
    
    world_name = "BARN/world_%d.world" %(args.world_idx)
    print(">>>>>>>>>>>>>>>>>> Loading Gazebo Simulation with %s <<<<<<<<<<<<<<<<<<" %(world_name))
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')
    
    launch_file = join(base_path, 'launch', 'gazebo_launch.launch')
    world_name = join(base_path, "worlds", world_name)
    
    gazebo_process = subprocess.Popen([
        'roslaunch',
        launch_file,
        'world_name:=' + world_name,
        'gui:=' + ("true" if args.gui else "false")
    ])
    
    rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
    rospy.set_param('/use_sim_time', True)
    
    # GazeboSimulation provides useful interface to communicate with gazebo simulation
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)
    
    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])
    
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True
    
    # check whether the robot is reset, the collision is False
    while compute_distance(init_coor, curr_coor) > 0.1 or collided:
        gazebo_sim.reset() # Reset to the initial position
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(1)
    
    launch_file = join(base_path, '..', args.navigation_stack)
    # Your launch file should take in two arguments (goal_x and goal_y) that specifies the goal
    nav_stack_process = subprocess.Popen([
        'roslaunch',
        launch_file,
        "goal_x:=%.2f" %GOAL_POSITION[0],
        "goal_y:=%.2f" %GOAL_POSITION[1],
    ])
    
    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)

    
    # check whether the robot started to move
    while compute_distance(init_coor, curr_coor) < 0.1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)
    
    # start navigation, check position, time and collision
    start_time = curr_time
    collided = False
    
    while compute_distance(goal_coor, curr_coor) > 0.3 and not collided and curr_time - start_time < 100:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(0.01)
        
    print(">>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    if collided:
        status = "collided"
    elif curr_time - start_time >= 100:
        status = "timeout"
    else:
        status = "succeeded"
    print("Navigation %s with time %.4f (s)" %(status, curr_time - start_time))
    
    gazebo_process.terminate()
    nav_stack_process.terminate()
