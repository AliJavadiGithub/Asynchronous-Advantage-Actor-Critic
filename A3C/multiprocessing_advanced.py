import torch.multiprocessing as mp
import os
import gym

os.environ['OMP_NUM_THREADS'] = '1'

class ParallelEnv:
    def __init__(self, env_id, num_threads):
        threads = [str(i) for i in range(num_threads)]
        self.ps = [mp.Process(target=worker, args=(thread, env_id)) for thread in threads]
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]  

def worker(name, env_id):
    env = gym.make(env_id)
    episode, max_eps, scores = 0, 10, []
    while episode < max_eps:
        obs = env.reset()
        score, done = 0, False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, truncated, info = env.step(action)  
            score += reward
            obs = obs_
        scores.append(score)
        print('episode {} process {} score {:.2f}'
                .format(episode, name, score))
        episode += 1

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # mp.set_start_method('spawn') works on Linux and MacOs
    env_id = 'CartPole-v0'
    n_threads = 4
    env = ParallelEnv(env_id=env_id, num_threads=n_threads)