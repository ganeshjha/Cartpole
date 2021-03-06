# Cartpole
OpenAI's CartPole problem in Reinforcement Learning
![cartpole_icon](https://user-images.githubusercontent.com/7412957/50371722-67a9bb00-0575-11e9-815d-e934495093b5.png)

Introduction
OpenAI's package, Gym , is a popular packages for comparing different reinforcement algorithms. In this project, there were different approaches of q-learning to solve the problem of cartpole problem: function approximation, deep q-value network, double q-value network, and experience replay for neural networks. Function approximation was done by Gregory, Deep Q Network (DQN) and experience replay was done by Ganesh, and Double Q Network (DDQN) and its experience replay was done by Jaehoon. 


DEEP Q-LEARNING
I implemented the Deep Q learning method using the keras framework that uses tensorflow as the backend. DQN is a RL technique that is aimed at choosing the best action for given circumstances (observation). Each possible action for each possible observation has its Q value, where 'Q' stands for a quality of a given move. The neural network I implemented had 4 input neurons with one hidden layer. I was getting pretty weird results in the beginning and then I came to find out that the reason was having a constant epsilon value. Then I tried decaying the epsilon value and implemented experience-replay technique which simply just performed well in around 900 episodes. Experience replay is a key technique behind many recent advances in deep reinforcement learning. Allowing the agent to learn from earlier memories can speed up learning and break undesirable temporal correlations.

![cartpole_output](https://user-images.githubusercontent.com/7412957/50371721-64aeca80-0575-11e9-92ee-f6c743ef0305.gif)

The roadblocks I encountered were in training the neural network and experimenting with the hyper-parameters. Till now I had implemented neural networks in predictive modelling only but using those concepts, dynamics and implementing in Reinforcement Learning was completely new.  I had to experiment various values as an input to parameters in order to train the networks.
