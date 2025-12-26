import numpy as np
import os
from procgen import ProcgenEnv
ENV_NAME = "coinrun"
DISTRIBUTION_MODE = "hard"
TOTAL_FRAMES = 5000 # smaller than the 10M transitions as used in the paper; this training set merely exists as a test of function
DATA_DIR = "coinrun_data"
env = ProcgenEnv(num_envs=1, env_name=ENV_NAME, distribution_mode=DISTRIBUTION_MODE, num_levels = 0) # note, nums_level = 0 implies infinite/persistently unique levels. 10000 levels were used as per appendix F.1, but, 0 is functionally equivalent. many levels is used to prevent overfitting.
obs = env.reset() # initialization of env - first 'frame'.
