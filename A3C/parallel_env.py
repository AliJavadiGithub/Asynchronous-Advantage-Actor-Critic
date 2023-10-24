import torch.multiprocessing as mp
from worker import worker
from shared_adam import SharedAdam
from actor_critic import ActorCritic

class ParallelEnv:
    def __init__(self, env_id, global_idx, input_shape,
                 n_actions, num_threads):
        names = [str(i) for i in range(num_threads)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        self.ps = [mp.Process(target=worker, 
                              args=(name, input_shape, n_actions,
                                    global_actor_critic, global_optim, env_id,
                                    num_threads, global_idx)) 
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]  