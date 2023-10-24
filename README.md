# RL-Algorithms-Implementations
# Simple implementation of Reinforcement Learning (A3C) using Pytorch

This is a toy example of using multiprocessing in Python to asynchronously train a
neural network to play discrete action [Pong](https://gym.openai.com/envs/PongNoFrameskip-v4/).
The asynchronous algorithm I used is called [Asynchronous Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783.pdf) or A3C.


## What are the main focuses in this implementation?

* Pytorch + multiprocessing (NOT threading) for parallel training
* Discrete action environments
* To be simple and easy to dig into the code (less than 200 lines)

## Reason of using [Pytorch](http://pytorch.org/) instead of [Tensorflow](https://www.tensorflow.org/)

Both of them are great for building your customized neural network. But to work
with multiprocessing, Tensorflow is not that great due to its low compatibility with multiprocessing.
However, the distributed version is for cluster computing which I don't have.
When using only one machine, it is slower than threading version I wrote.

Fortunately, Pytorch gets the [multiprocessing compatibility](http://pytorch.org/docs/master/notes/multiprocessing.html).

BTW, if you are interested to learn Pytorch, [there](https://github.com/MorvanZhou/PyTorch-Tutorial)
 is my simple tutorial code with many visualizations. I also made the tensorflow tutorial (same as pytorch) available in [here](https://github.com/MorvanZhou/Tensorflow-Tutorial).

## Codes & Results

* [shared_adam.py](/A3C/shared_adam.py): optimizer that shares its parameters in parallel
* [utils.py](/A3C/utils.py): useful function that can be used more than once

Pong result
![Pong](/results/A3C_pong_final_4threads.png)

## Dependencies

* pytorch >= 0.4.0
* numpy
* gym
* matplotlib
