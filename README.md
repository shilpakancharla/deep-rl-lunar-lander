# Applications of Reinforcement Learning: Lunar Lander Simulation

## Abstract

The purpose of the following reinforcement learning experiment is to investigate optimal parameter values for deep Q-learning (DQN) on the Lunar Lander problem provided by OpenAI Gym. The \texttt{LunarLander-v2} is an environment with uncertainty and this investigation explores optimal parameters that will maximize the mean reward over 400 episodes or less. A deep learning network is designed for the agent and various reinforcement learning parameters are used to carry out the simulation. Through the use of a neural network with two hidden layers, the agent was able to converge to a mean average reward score of 200 with $\epsilon = 0.9$, $\epsilon$-decay $=0.995$, $\alpha = 0.001$, and $\gamma = 0.99$ in a little over 250 episodes. A comparative analysis between different parameters used is also performed. The results and the architecture of the model used from this experiment are also compared to other similar experiments that employ the DQN method for the Lunar Lander problem.



