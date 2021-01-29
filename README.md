# View Pose Planning with Reinforcement Learning

## Installation

#### Requirements
- Ubuntu 20.04 (Virtual Machine or native)
- ROS Noetic. Follow [this](http://wiki.ros.org/noetic/Installation/Ubuntu) installation instructions

#### Install dependencies

- Set python3 as default ```sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10```.
- Install ```pip sudo apt install python3-pip```
- Recommended, but not necessary: catkin_tools https://catkin-tools.readthedocs.io/en/latest/installing.html. Install missing libraries with pip.
- ```sudo apt-get install python3-catkin-tools```
- ```sudo apt install python3-catkin-lint python3-pip```
- ```pip3 install osrf-pycommon```
- Install rosdep as described [here](http://wiki.ros.org/rosdep)

#### ROS Workspace
- Create a ROS workspace in your home directory by executing ```mkdir -p catkin_ws/src```, ```cd catkin_ws``` and ```catkin init```
- Clone this repository via ```git clone https://github.com/christianlandgraf/rl_viewplanning.git``` into ~/catkin_ws/src
- Clone the following Github repositiories to ~/catkin_ws/src:
```
git clone https://github.com/christianlandgraf/universal_robot.git
git clone https://github.com/christianlandgraf/Universal_Robots_ROS_Driver.git
```
- Install ROS dependencies  from source 
```
cd catkin_ws
rosdep install --from-paths src --ignore-src -r -y
```
- Install ROS dependencies : ```rosdep install --from-paths src --ignore-src -r -y```
- Install the following ROS binary packages
```
sudo apt install ros-noetic-moveit ros-noetic-moveit-resources ros-noetic-geometry2 ros-noetic-graph-msgs ros-noetic-velodyne-simulator
```
- Additionally, clone the following to your workspace and build it from source. If they are available as binary package, install them as described above

```
cd catkin_ws/src
git clone https://github.com/ros-planning/moveit_visual_tools.git
git clone https://github.com/gavanderhoorn/industrial_robot_status_controller.git
git clone https://github.com/ros-industrial/industrial_core.git
git clone https://bitbucket.org/DataspeedInc/velodyne_simulator.git
git clone https://github.com/PickNikRobotics/rviz_visual_tools.git
```

- We experienced issues with the industrial core package in Noetic. Ignore the folder industrial_trajectory_filters by
```
cd catkin_ws/src/industrial_core/industrial_trajectory_filters
touch CATKIN_IGNORE
```

- In universal_robot, switch the branch
```
cd catkin_ws/src/universal_robot
git checkout calibration_devel
```
- Build the workspace with ```catkin build```.
- Source the workspace ```source devel/setup.bash```

#### OpenAi

Clone gym to your home folder and install it via:
```
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e .
pip3 install git-python
```

#### Open3D

Follow the inststructions at http://www.open3d.org/.
- ```pip3 install open3d```

#### PCL

```
sudo add-apt-repository ppa:sweptlaser/python3-pcl
sudo apt update
sudo apt install libpcl-dev python3-pcl
```

#### Stable Baselines

- Stable Baselines 3: ```pip3 install stable-baselines3```, see the [docs](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

#### Tensorboard

Install tensorboard via ```pip install tensorboard```. 
Open tensorboard by using the following command in a new terminal

```
tensorboard --logdir=<current log folder>
```




## Troubleshooting:

Error | Description 
--- | ---
could not find "<xyz>Config.cmake" during build | it usually means package <xyz> is missing. Either try to find the package with apt and autocompletion, e.g. ```sudo apt install ros-noetic-xyz```. If this doesn't work either, search for the package and clone it to your src folder from Github/Gitlab/Bitbucket etc. Then try to build it from source with ```catkin build xyz```.
Cannot find package when launching | The most common error in ROS is to forget to source your workspace with, e.g. ```source ~/catkin_ws/devel/setup.bash```. Put it in your ~/.bashrc file: ```source ~/catkin_ws/devel/setup.bash```
Gazebo does not close | Copy the following command in your ~/.bashrc to directly kill Gazebo in the terminal: ```alias killgazebo="killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient"```. When restarting the environment, kill all gazebo processes with ```killgazebo```


## Overview

Main scripts:

- ipa_kifz_env.py (robot environment)
    - location:  ```automeasure_ros/project_kifz/openai_ros/src/openai_ros/robot_envs```
    - plan_pose: Use Python Moveit (Env)
    - get_point_cloud: Process Point Cloud data and Conversion to Open3d format (Env)
- ipa_kifz_viewplannning (task environment)
    - location:  ```automeasure_ros/project_kifz/openai_ros/src/openai_ros/task_envs```
    - _set_action: Go to next position (Task)
    - _is_done: Check if episode should end (Task)
    - _compute_reward: Compute reward (Task)
    - _get_obs: Combine Observation (Task)
- start_training_qlearn.py (main script, "RL agent(?)")
    - location: ```automeasure_ros/project_kifz/ipa_kifz_viewplanning/scripts/qlearn```
    - Set state from observations in task environment
    - Interact between task environment and RL algorithm
- qlearn.py (RL algorithm)
    - location: ```automeasure_ros/project_kifz/ipa_kifz_viewplanning/scripts/qlearn```
- ipa_kifz_viewplannning.yaml (Task-related configuration)
    - location:  ```automeasure_ros/openai_ros/openai_ros/src/openai_ros/task_envs/config```
- ipa_kifz_viewplannning_qlearn.yaml (RL Algorithm-related configuration)
    - location: ```automeasure_ros/project_kifz/ipa_kifz_viewplanning/config```


Other components and scripts:
- visualization.py
    - location: ```automeasure_scripts/```
    - Visualizes the reward, can also be done via TensorBoard
- generate_dataset.py
    - location: ```automeasure_ros/project_kifz/ipa_kifz_viewplanning/scripts/dataset_generation```
    - Simple generate random poses and save the result (needs some adjustments in ipa_kifz_viewplanning)
    - Can be started via ```roslaunch ipa_kifz_viewplanning start_training.launch algorithm:=generate_dataset```
- viewpoint_sampler.py
    - location: ```automeasure_ros/project_kifz/ipa_kifz_viewplanning/scripts/viewpoint_sampling```
    - Proposes random points to save as pre-sampled viewpoints
    - Can be started via ```roslaunch ipa_kifz_viewplanning start_training.launch algorithm:=viewpoint_sampling```


## Usage

1. Start simulation environment: ```roslaunch ipa_kifz_viewplanning ipa_kifz_gazebo.launch``` with following options:
    - ```gazebo_gui:=False``` If Gazebo should run in headless mode
    - ```rviz_gui:=False``` If RViz visualiuation should run in parellell
    - ```sensor_only:=True``` If the robot layout and its semantics should be neglected.
2. Start the learning procedure: ```roslaunch ipa_kifz_viewplanning start_training.launch```
    - ```algorithm:=<algorithm>``` The RL Algorithm to be used, currently available: qlearn, ppo.
    Other options are generate_dataset (to sample and save point clouds for point cloud representation experiments)
    and viewpoint_sampling (to sample and save a discrete number of viewpoints for subsequent planning)


## TODO