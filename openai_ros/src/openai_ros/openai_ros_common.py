#!/usr/bin/env python
import gym
from .task_envs.task_envs_list import RegisterOpenAI_Ros_Env
import roslaunch
import rospy
import rospkg
import os
import git
import sys
import subprocess


def StartOpenAI_ROS_Environment(task_and_robot_environment_name):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn(
        "Env: {} will be imported".format(task_and_robot_environment_name))
    result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=10000)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..." +
                      str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env


def StartOpenAI_ROS_Environment_Parallel(task_and_robot_environment_name,
                                         env_id):
    """
    Does the same as the previous function but allows to start multiple Environments in Parallel
    """
    rospy.logwarn(
        "Env: {} will be imported".format(task_and_robot_environment_name))
    result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=10000)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..." +
                      str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name,
                       env_id=int(str(env_id)))
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None
    return env


class ROSLauncher(object):
    def __init__(self, rospackage_name, launch_file_name, arguments=""):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()

        # Check Package Exists
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package FOUND...")
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT FOUND, lets Download it...")
            rospy.logerr("Package " + rospackage_name + " NOT FOUND")

        # If the package was found then we launch
        if pkg_path:
            rospy.loginfo(">>>>>>>>>>Package found in workspace-->" +
                          str(pkg_path))
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name==" +
                          str(path_launch_file_name))

            roslaunch_command = "roslaunch {0} {1} {2}".format(
                rospackage_name, launch_file_name, arguments)
            command = roslaunch_command
            rospy.logwarn("Launching command=" + str(command))

            p = subprocess.Popen(command, shell=True)

            state = p.poll()
            if state is None:
                rospy.loginfo("process is running fine")
            elif state < 0:
                rospy.loginfo("Process terminated with error")
            elif state > 0:
                rospy.loginfo("Process terminated without error")

            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS apckage ==>" + \
                str(rospackage_name)