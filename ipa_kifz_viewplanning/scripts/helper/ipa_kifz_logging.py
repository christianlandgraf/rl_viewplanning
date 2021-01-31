#!/usr/bin/env python3
import os
import pandas as pd
from shutil import copyfile
import rospy
import rospkg


class Logger():
    def __init__(self, timestamp, log_freq=1):

        # Set variables
        self.algorithm = rospy.get_param("algorithm")
        self.timestamp = timestamp
        self.log_freq = log_freq

        # Set directories
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path("ipa_kifz_viewplanning")
        self.openai_ros_pkg_path = rospack.get_path("openai_ros")
        self.training_directory = os.path.join(self.pkg_path,
                                               "training_results",
                                               self.algorithm + "_results")
        self.log_directory = os.path.join(self.training_directory,
                                          str(self.timestamp))

        # Choose file names
        self.training_log = "training_log.json"
        self.param_log = "model_parameters.json"
        self.task_log = "task_parameters.json"

        # Create directories and data framews
        if not os.path.exists(self.training_directory):
            os.makedirs(self.training_directory)

        # Create or load directory and log
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            self.df = pd.DataFrame(
                columns=["episode_rewards", "episode_poses"])
        else:
            #TODO Load directory content
            pass

        # Helper Variables
        self.episode_counter = 0
        self.current_poses = []

    def save_params(self):
        algorithm_config = os.path.join(
            self.pkg_path, "config",
            "ipa_kifz_viewplanning_" + self.algorithm + "_params.yaml")
        task_config = os.path.join(self.openai_ros_pkg_path, "src",
                                   "openai_ros", "task_envs", "config",
                                   "ipa_kifz_viewplanning.yaml")
        dest_algorithm_config = os.path.join(
            self.log_directory,
            "ipa_kifz_viewplanning_" + self.algorithm + "_params.yaml")
        dest_task_config = os.path.join(self.log_directory,
                                        "ipa_kifz_viewplanning.yaml")
        copyfile(algorithm_config, dest_algorithm_config)
        copyfile(task_config, dest_task_config)

    def load_params(self):
        pass

    def save_model(self, model):
        if self.algorithm == "qlearn":
            # Save Q Table
            file_name = os.path.join(self.log_directory, "model.json")
            with open(file_name, "w") as file:
                file.write(str(model))
        else:
            # Save Stable Baselines Model
            model.save(
                os.path.join(self.log_directory, "ipa_kifz_viewplanning-v0"))
            pass

        self.episode_counter = 0
        return

    def load_model(self):
        pass

    def log_pose(self, pose):
        self.current_poses.append(pose)
        return

    def log_episode(self, episode_reward, episode_poses=None, model=None):

        # Save reward and chosen poses
        if episode_poses is not None:
            self.df.loc[len(self.df)] = [episode_reward, episode_poses]
        else:
            self.df.loc[len(self.df)] = [episode_reward, self.current_poses]
            self.current_poses = []

        # Regularly save dataframe and model
        self.episode_counter += 1
        if (self.episode_counter % self.log_freq) == 0:
            self.df.to_json(os.path.join(self.log_directory,
                                         self.training_log),
                            orient="index")
            if model is not None:
                self.save_model(model)

        return

    def save_env(self):
        # Other parameters to be saved:
        # - workpiece
        # - workpiece pose
        # - sensor configuration
        # - plots
        pass
