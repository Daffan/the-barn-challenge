import numpy as np

BASE_WORLD_PATH = "scripts/base_world.world"

def make_head(name, time_interval):
  return "\n\
#include <gazebo/gazebo.hh>\n\
#include <ignition/math.hh>\n\
#include <gazebo/physics/physics.hh>\n\
#include <gazebo/common/common.hh>\n\
#include <stdio.h>\n\
\n\
namespace gazebo\n\
{\n\
  class %s : public ModelPlugin\n\
  {\n\
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)\n\
    {\n\
      // Store the pointer to the model\n\
      this->model = _parent;\n\
\n\
        // create the animation\n\
        gazebo::common::PoseAnimationPtr anim(\n\
              // name the animation '%s',\n\
              // make it last 10 seconds,\n\
              // and set it on a repeat loop\n\
              new gazebo::common::PoseAnimation(\"%s\", %.2f, true));\n\
\n\
        gazebo::common::PoseKeyFrame *key;\n\
" %(name, name, name, time_interval)

def make_tail(name):
  return "\n\
        // set the animation\n\
        _parent->SetAnimation(anim);\n\
    }\n\
\n\
    // Pointer to the model\n\
    private: physics::ModelPtr model;\n\
\n\
    // Pointer to the update event connection\n\
    private: event::ConnectionPtr updateConnection;\n\
  };\n\
\n\
  // Register this plugin with the simulator\n\
  GZ_REGISTER_MODEL_PLUGIN(%s)\n\
}\n\
" %(name)

def make_waypoint(time, x, y):
    return "\n\
        key = anim->CreateKeyFrame(%.2f);\n\
        key->Translation(ignition::math::Vector3d(%.2f, %.2f, 0));\n\
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));\n\
" %(time, x, y)

def make_CMakeLists(name_list):
    s = "\n\
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)\n\
\n\
project(animated_box)\n\
\n\
# Find packages\n\
\n\
find_package(gazebo REQUIRED)\n\
\n\
list(APPEND CMAKE_CXX_FLAGS '${GAZEBO_CXX_FLAGS}')\n\
\n\
# include appropriate directories\n\
include_directories(${GAZEBO_INCLUDE_DIRS})\n\
link_directories(${GAZEBO_LIBRARY_DIRS})\n\
\n\
# Create libraries and executables\n\
"
    for n in name_list:
        s += "\n\
add_library(%s SHARED %s.cc)\n\
target_link_libraries(%s ${GAZEBO_LIBRARIES})\n\
" %(n, n, n)
    return s

def make_moving_model(plugin):
    name = plugin.split(".so")[0]
    return "\n\
    <model name='%s'>\n\
      <static>1</static>\n\
      <pose frame=''>-0.225000 0.075000 0.000000 0.000000 0.000000 0.000000</pose>\n\
      <link name='link'>\n\
        <inertial>\n\
          <mass>1</mass>\n\
          <inertia>\n\
            <ixx>0.145833</ixx>\n\
            <ixy>0</ixy>\n\
            <ixz>0</ixz>\n\
            <iyy>0.145833</iyy>\n\
            <iyz>0</iyz>\n\
            <izz>0.125</izz>\n\
          </inertia>\n\
        </inertial>\n\
        <collision name='collision'>\n\
          <geometry>\n\
            <cylinder>\n\
              <radius>0.075000</radius>\n\
              <length>0.8</length>\n\
            </cylinder>\n\
          </geometry>\n\
          <max_contacts>10</max_contacts>\n\
          <surface>\n\
            <contact>\n\
              <ode/>\n\
            </contact>\n\
            <bounce/>\n\
            <friction>\n\
              <torsional>\n\
                <ode/>\n\
              </torsional>\n\
              <ode/>\n\
            </friction>\n\
          </surface>\n\
        </collision>\n\
        <visual name='visual'>\n\
          <geometry>\n\
            <cylinder>\n\
              <radius>0.075000</radius>\n\
              <length>0.8</length>\n\
            </cylinder>\n\
          </geometry>\n\
	    <material>\n\
              <ambient>0.152000 0.379000 0.720000 1</ambient>\n\
	      <diffuse>0.152000 0.379000 0.720000 1</diffuse>\n\
          </material>\n\
          <material>\n\
            <script>\n\
              <name>Gazebo/Grey</name>\n\
              <uri>file://media/materials/scripts/gazebo.material</uri>\n\
            </script>\n\
          </material>\n\
        </visual>\n\
        <self_collide>0</self_collide>\n\
        <kinematic>0</kinematic>\n\
        <gravity>1</gravity>\n\
      </link>\n\
      <plugin name='%s' filename='%s'/>\n\
    </model>\n\
" %(name, name, plugin)

def sample_waypoints(direction, min_speed, max_speed, idx):
    assert direction in DIRECTIONS, "direction %s not defined!" %(direction)
    x1, y1 = BOTTOM_LEFT
    x2, y2 = TOP_RIGHT
    if direction == "B2T":
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y1
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y2
    elif direction == "T2B":
        end_x = np.random.random() * (x2 - x1) + x1
        end_y = y1
        start_x = np.random.random() * (x2 - x1) + x1
        start_y = y2
    elif direction == "R2L":
        start_x = x1
        start_y = np.random.random() * (y2 - y1) + y1
        end_x = x2
        end_y = np.random.random() * (y2 - y1) + y1
    elif direction == "L2R":
        end_x = x1
        end_y = np.random.random() * (y2 - y1) + y1
        start_x = x2
        start_y = np.random.random() * (y2 - y1) + y1

    distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5
    speed = np.random.uniform(min_speed, max_speed)
    time = distance / speed
    return "%s_%d_%d" %(direction, int(speed * 100), idx), [(0, start_x, start_y), (time, end_x, end_y)]

BOTTOM_LEFT = (-4.5, 5)
TOP_RIGHT = (0, 9.5)

DIRECTIONS = ["R2L", "L2R", "T2B", "B2T"]

if __name__ == "__main__":
    import argparse
    import subprocess
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="jackal_helper/worlds/BARN")
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--min_speed', type=float, default=0.1)
    parser.add_argument('--max_speed', type=float, default=1)
    parser.add_argument('--min_object', type=int, default=2)
    parser.add_argument('--max_object', type=int, default=6)
    parser.add_argument('--start_idx', type=int, default=300)
    parser.add_argument('--n_worlds', type=int, default=100)
    parser.add_argument('--rebuild_plugin', action="store_true")
    parser.add_argument('--plugins_per_direction', type=int, default=20)
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    #1 build plugins
    plugins_dir = "jackal_helper/plugins"
    if args.rebuild_plugin or not os.path.exists(plugins_dir):
        np.random.seed(args.seed)
        os.mkdir(plugins_dir)
        name_list = []
        for d in DIRECTIONS:
            for i in range(args.plugins_per_direction):
                name, waypoints = sample_waypoints(d, args.min_speed, args.max_speed, i)
                name_list.append(name)
                fs = make_head(name, waypoints[-1][0] - waypoints[0][0])
                for wp in waypoints:
                    fs += make_waypoint(*wp)
                fs += make_tail(name)
                with open(os.path.join(plugins_dir, name + ".cc"), "w") as f:
                    f.writelines(fs)
        
        cmake_fs = make_CMakeLists(name_list)
        with open(os.path.join(plugins_dir, "CMakeLists.txt"), "w") as f:
            f.writelines(cmake_fs)
        
        os.mkdir(os.path.join(plugins_dir, "build"))
    
        wd = os.getcwd()
        os.chdir(os.path.join(wd, plugins_dir, "build"))
        subprocess.run(["cmake", ".."])
        subprocess.call("make")       
        os.chdir(wd)
    
    #2 create .world files
    np.random.seed(args.seed + 1)
    plugins_build_dir = os.path.join(plugins_dir, "build")
    plugins = [f for f in os.listdir(plugins_build_dir) if f.endswith(".so")]
    with open(BASE_WORLD_PATH, "r") as f:
        ss = f.read()
        part1 = ss.split("TOKEN")[0]
        part2 = ss.split("TOKEN")[1]
        
    for i in range(args.n_worlds):
        mid = ""
        for j in range(np.random.randint(args.min_object, args.max_object)):
            plugin = np.random.choice(plugins)
            mid += make_moving_model(plugin)
            
        with open(os.path.join(args.save_dir, "world_%d.world" %(i + args.start_idx)), "w") as f:
            f.write(part1 + mid + part2)