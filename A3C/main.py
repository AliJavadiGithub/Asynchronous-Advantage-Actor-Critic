import torch.multiprocessing as mp
import os
from parallel_env import ParallelEnv

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    env_id = 'PongNoFrameskip-v4'
    n_threads = 8
    n_actions = 6
    input_shape = [4, 42, 42]
    env = ParallelEnv(env_id=env_id, num_threads=n_threads, 
                 n_actions=n_actions, global_idx=global_ep, 
                 input_shape=input_shape)