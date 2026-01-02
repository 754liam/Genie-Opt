# implementation of appendix F.1
import numpy as np
import os
from procgen import ProcgenEnv
ENV_NAME = "coinrun"
DISTRIBUTION_MODE = "hard"
TOTAL_FRAMES = 5000 # smaller than the 10M transitions as used in the paper; this training set merely exists as a test of function
DATA_DIR = "coinrun_data"
env = ProcgenEnv(num_envs=1, env_name=ENV_NAME, distribution_mode=DISTRIBUTION_MODE, num_levels = 0) # note, nums_level = 0 implies infinite/persistently unique levels. 10000 levels were used as per appendix F.1, but, 0 is functionally equivalent. many levels is used to prevent overfitting.
obs = env.reset() # initialization of env - first 'frame'.

all_frames = []
for step in range(TOTAL_FRAMES):
    action = np.array([env.action_space.sample()]) #input for the game engine. Note that action_space is the set of all valid operations, and .sample() returns a random operation such that it is expressed in integer form
    obs, reward, done, info = env.step(action) # Note that env.step is a parallel processor - it only accepts lists of numbers, hence why the former is a np array. the action as given in the line above is passed into the engine. A tuple from such is returned giving us the Numpy array of the new image, a reward if applicable, a boolean representing completeness, and a debug dict.
    all_frames.append(obs['rgb'].astype(np.uint8)) # memory management! the Numpy array contained in obs defaults to 32-bit floating point numbers, but, this is functionally similar to unsigned 8 bit integers. This reduces memory for this part by 4x. 
dataset = np.stack(all_frames) # all_frames now is a list of 5000 pointers - np.stack() places these all in one 4D tensor. massive speed optimization.
os.makedirs(DATA_DIR, exist_ok=True) 
save_path = os.path.join(DATA_DIR, "training_data.npy")
np.save(save_path, dataset)  # saves the dataset on the hard drive
print("Data saved successfully.") # Useful for Docker run debugging.

