<p align="center">
  <img width = "100%" src='res/BARN_Challenge.png' />
  </p>

--------------------------------------------------------------------------------

# ICRA 2022 BARN Challenge

## Requirements
If you run it on a local machine without containers:
* ROS version at least Kinetic
* CMake version at least 3.0.2
* Python version at least 3.6
* Python packages: defusedxml, rospkg, netifaces, numpy

If you run it in Singularity containers:
* Go version at least 1.13
* Singularity version at least 3.6.3

The requirements above are just suggestions. If you run into any issue, please contact organizers for help (zfxu@utexas.edu).

## Installation
Follow the instructions below to run simulations on your local machines.

1. Create a virtual environment (we show examples with python venv, you can use conda instead)
```
apt -y update; apt-get -y install python3-venv
python3 -m venv /<YOUR_HOME_DIR>/nav_challenge
export PATH="/<YOUR_HOME_DIR>/nav_challenge/bin:$PATH"
```

2. Install Python dependencies
```
pip3 install defusedxml rospkg netifaces numpy
```

3. Create ROS workspace
```
mkdir -p /<YOUR_HOME_DIR>/jackal_ws/src
cd /<YOUR_HOME_DIR>/jackal_ws/src
```

4. Clone this repo and required ros packages
```
git clone https://github.com/Daffan/nav-competition-icra2022.git
git clone https://github.com/jackal/jackal.git
git clone https://github.com/jackal/jackal_simulator.git
git clone https://github.com/jackal/jackal_desktop.git
git clone https://github.com/utexas-bwi/eband_local_planner.git
```

5. Install ROS package dependencies
```
cd ..
source /opt/ros/<YOUR_ROS_VERSION>/setup.bash
rosdep init; rosdep update
rosdep install -y --from-paths . --ignore-src --rosdistro=melodic
```

6. Build the workspace
```
source devel/setup.bash
catkin_make
```

Follow the instruction below to run simulations in Singularity containers.

1. Follow this instruction to install Singularity: https://sylabs.io/guides/3.0/user-guide/installation.html. Singularity version >= 3.6.3 is required to successfully build the image!

1. Build Singularity image (sudo access required)
```
sudo singularity build --notest nav_competition_image.sif Singularityfile.def
```

## Run Simulations
If you run it on your local machines: (the example below runs [move_base](http://wiki.ros.org/move_base) with DWA local planner in world 0)
```
python3 run.py \
--world_idx 0 \
--navigation_stack jackal_helper/launch/move_base_DWA.launch
```

If you run it in a Sinularity container:
```
./singularity_run.sh /path/to/image/file python3 run.py \
--world_idx 0 \
--navigation_stack jackal_helper/launch/DWA.launch
```

A successful run should print the episode status (collided/succeeded/timeout) and the time cost in second:
> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
> Navigation collided with time 27.2930 (s)

> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
> Navigation succeeded with time 29.4610 (s)


> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
>Navigation timeout with time 100.0000 (s)

## Test your own navigation stack
We currently don't provide a lot of instructions or a standard API for implementing the navigation stack, but we might add more in this section depending on people's feedback. If you are new to the ROS or mobile robot navigation, we suggest checking [move_base](http://wiki.ros.org/move_base) which provides basic interface to manipulate a robot.

We recommand building your navigation stack as a separate ROS package and upload it on github. Once you have you own github repo, clone it under the path `/<YOUR_HOME_DIC>/jackal_ws/src`, then rebuid the work space. if you use Singularity container, add the command to clone your repo before line 18 in `Singularityfile.def`

Your navigation stack should be called with a single launch file (similar to `jackal_helper/launch/move_base_DWA.launch` or `jackal_helper/launch/move_base_eband.launch`). The launch file takes two arguments: `goal_x` and `goal_y` that specifies the relative goal location in the world frame.

You can then test your own navigation stack as follow:
```
python3 run.py \
--world_idx 0 \
--navigation_stack ../<YOUR_REPO>/<YOUR_LAUNCH_FILE>
```

## Submission
TBD...
