"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""

import argparse
import os, time, cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from scipy.spatial.transform import Rotation as R

import omnigibson as og
from omnigibson import example_config_path
from omnigibson.macros import gm
from omnigibson.robots.robot_base import BaseRobot
from gym import spaces

from stable_baselines3.common.utils import set_random_seed


from robo_dataset import TextImageDataset

try:
    import gym
    import torch as th
    import torch.nn as nn
    import tensorboard
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

except ModuleNotFoundError:
    og.log.error("torch, stable-baselines3, or tensorboard is not installed. "
                 "See which packages are missing, and then run the following for any missing packages:\n"
                 "pip install stable-baselines3[extra]\n"
                 "pip install tensorboard\n"
                 "pip install shimmy>=0.2.1\n"
                 "Also, please update gym to >=0.26.1 after installing sb3: pip install gym>=0.26.1")
    exit(1)


# We don't need object states nor transitions rules, so we disable them now, and also enable flatcache for maximum speed
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True

from torchvision import models

class PandaEnv(og.Environment):
    def __init__(self, configs):
        super().__init__(configs)
        self.robot = self.robots[0]
        self.ee_pos = self.robot.get_eef_position()
        
        # target is an offset from the current position
        # for now, task is to move the EE to the target position
        self.target_position = self.ee_pos + np.array([0.1, 0.1, 0.1])
        
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # get robot EE position
        self.ee_pos = self.robot.get_eef_position()
        
        depth = np.random.random(2)
        language_embedding = np.random.random(2)
        extended_obs = np.concatenate((depth, language_embedding))
        custom_reward = self.custom_reward(obs, reward, done)

        return obs, custom_reward, done, info
        
    def reset(self):
        return super().reset()
    
    def custom_reward(self, obs, reward, done):
        dist = np.linalg.norm(self.ee_pos - self.target_position)
        reward = -dist
        
        if dist < 0.01:
            reward = 10
            done = True
        return reward
    
def orient_forward_facing(robot: BaseRobot, env: og.Environment):
    action = np.zeros(robot.action_dim)
    for i in range(50):
        action[4] -= 0.1
        env.step(action)
    time.sleep(2)
    
    
def exp(env, env_robot, target_pos, target_pos_back, original_pos_back, original_pos, dataset, item, start_with_move_to=True,
        total_trials=0):
    for idx in range(4):
        command = "move to" if start_with_move_to else "move away from"
        dataset = run_traj(env, env_robot, target_pos, dataset, command, item)
        time.sleep(1)
        
        command = 'move away from' if start_with_move_to else 'move to'
        dataset = run_traj(env, env_robot, target_pos_back, dataset, command, item)
        time.sleep(1)
        env.reset()
        
        orient_forward_facing(env_robot, env)
        
        target_pos = original_pos + (np.random.random(3) * 0.02)
        target_pos_back = original_pos_back + (np.random.random(3) * 0.05)
        
        # there is a bug in trial 0, so we skip it
        if total_trials != 0:
            # skip trial 0
            dataset.save(total_trials)
        else: 
            dataset.data = []
        total_trials += 1
        
    env.reset()
    orient_forward_facing(env_robot, env)
    return total_trials


def main():
    # Parse args
    # create tensorboard log dir
    tensorboard_log_dir = os.path.join("log_dir", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    seed = 0
    
    # Load config
    with open(f"{example_config_path}/panda_behavior.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure flattened obs and action space is used
    cfg["env"]["flatten_action_space"] = True
    cfg["env"]["flatten_obs_space"] = True

    # Only use RGB obs
    cfg["robots"][0]["obs_modalities"] = ["rgb"]
    env = PandaEnv(configs=cfg)

    # Set the seed
    set_random_seed(seed)
    env.reset()
    
    # wait a few seconds
    time.sleep(3)
    
    env.robots[0].reset()
    
    env_robot: BaseRobot = env.robots[0]
    
    # sensor name is first key
    sensor_name = list(env_robot.sensors.keys())[0]
    
    dataset = TextImageDataset()
    command = "move to"
    item = "chair"
    
    action = np.zeros(env_robot.action_dim)
    
    orient_forward_facing(env_robot, env)
    
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    ee_start_pos = env_robot.get_eef_position()
    
    chair_pos = ee_start_pos + np.array([0.0, 0.2, 0.2])
    table_pos = ee_start_pos + np.array([0.0, 0.2, 0.3])
    wall_pos = ee_start_pos + np.array([0.0, 0.3, 0.4])
    painting_pos = ee_start_pos + np.array([0.0, 0.0, 0.4])
    window_pos = ee_start_pos + np.array([0.0, -0.3, 0.4])
    
    poses = [window_pos, chair_pos, table_pos, wall_pos, painting_pos]
    items = ['window', 'chair', 'table', 'wall', 'painting']
    total_trials = 0
    
    for i in range(5):
        print("Experiment for ", items[i])
        # target position is near the object
        target_pos = poses[i]
        original_pos = target_pos
    
        # position back, away from object
        target_pos_back = ee_start_pos + np.array([0, 0, 0])
        original_pos_back = target_pos_back
        
        total_trials = exp(env, env_robot, target_pos, target_pos_back, original_pos_back, original_pos, dataset, items[i], start_with_move_to=True, total_trials=total_trials)
        
    env.close()
    
##################################################################################3
    
def run_traj(env, env_robot, target_pos, dataset, command, item):
    step = 0
    max_steps = -1
    sensor_name = list(env_robot.sensors.keys())[0]
    
    action = np.zeros(env_robot.action_dim)
    while step != max_steps:
        obs, reward, done, info = env.step(action=action)

        ee_pos = env_robot.get_eef_position()
        loss = target_pos - ee_pos
        
        sensor_frames = env_robot.get_vision_data()
        rgb = sensor_frames["rgb"]
        
        camera_pose = env_robot.sensors[sensor_name].get_position_orientation()
            
        pos, quat = camera_pose
        
        # resize image to 256 x 256
        rgb = cv2.resize(rgb, (256, 256))
        
        # pid with kp = 1, ki = 0, kd = 0
        # action = loss
        # first 3 values of actions are the EE position
        kp = 1
        print(len(dataset), "Length of dataset")
        
        # p controller
        action[:3] = (kp * loss)
        print(action[:3], "Action", command)
        dataset.add_sample(command, item, rgb, action[:3], (pos, quat))
        # add sample of other direction
        other_command = 'move away from' if command == 'move to' else 'move to'
        dataset.add_sample(other_command, item, rgb, -action[:3], (pos, quat))
        if np.linalg.norm(loss) < 0.03:
            print('task complete!')
            break
        step += 1
    return dataset


if __name__ == "__main__":
    main()
