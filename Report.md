## Project 1: Navigation

Author: Abhijeet Biswas

### Project Goal

Train an agent to navigate a large world and collect yellow bananas, while avoiding blue bananas

![](navigation.gif)

### Project details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


### Learning Algorithm

I have implememented  Value-based method called [Double Deep Q-Networks](https://arxiv.org/abs/1509.06461) for this project. Deep Q-Networks (DQN) predicts the Q values for each of the actions based on the current state using Neural Networks. We select the action with maximum Q value using epsilon-greedy algorithm. But it suffers from the problem of overestimating the action values under certain conditions. A minor change to the DQN algorithm resulting the Double DQN algorithm not only reduces the observed overestimations but also leads to much better performance on several games. Hence, I chose to implement the Double DQN.

Training the network:

-  We will randomly initialize our Agent with 2 DQNs, one is called main network and other is called target network.

-  We will update the parameters of the main and train network after fixed time steps. Meanwhile we will keep storing experiences in Replay Buffer from agent's interaction with environment. We will then sample a batch of transitions(experiences) from the replay buffer if it is not empty and size is greater than batch size, else we will keep generating experiences from agent interaction until the replay buffer size equals the batch size.

- Using the next states from the sampled experiences, we will run the main network in order to find the action that maximizes Q, $\arg \max_{a} Q(s_{t+1},a)$. We will use these actions and the corresponding next states and run the target network to calculate $Q(s_{t+1},\arg\max_{a}Q(s_{t+1},a))$.

- We then use the following equation for calculating the targets of the network: $y_{t}= r(s_{t},a_{t})+\gamma Q(s_{t+1},\arg\max_{a}Q(s_{t+1},a))$ where r is th reward and $\gamma$ is the discounting factor.

- We then update our main network using the current states as inputs, and with the targets as mentioned above using Mean Squared Loss. We then perform a soft update for our target network by copying the parameters by weighing them from the main network to the target network. 

- We then repeat the whole process till the environment is solved




### CODE

The code is written in Python 3.6.3 and uses PyTorch 0.4.0. I have used detailed comments to explain all the steps.

### Network Architecture

I have used two DQNs each with 4 layer Neural Network with 37, 64, 64 and 4 neurons in each layer respectively. The first layer takes in the state which has 37 dimension and the last layer outputs the Q values for each of the 4 actions. Intermediate layers are used to build more complex features from input that will produce better results without overfitting. I have used [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation  and Dropout with 0.1 probability after each layer except the last one.

#### Other Hyperparameters
- eps: Epsilon is used to select action using epsilon-greedy algorithm. Started with 1 and decayed after each episode using
     eps= eps/episode_number
     
- tau: Used For soft update of target network, constant value of 1e-3 was used

- BUFFER_SIZE: Replay buffer  size of 1e5 was used

- BATCH_SIZE: Minibatch size of  64  was used

- UPDATE_EVERY: Episode Interval for network update, the network was updated after every 4 episodes        



### Results

The environment got solved in 353 episodes by having average reward (13.03) more than 13 over 100 consecutive episodes. The below plot shows the reward received after each episode

![reward_plot.png](attachment:reward_plot.png)

### Future Ideas

1. Implement Convolutional Neural Network to directly learn from pixel values. Will also try fusing CNN features and state vectors to see if there is any improvement in training.
2. Use prioritized replay buffer to replay important transitions more frequently, and therefore learn more efficiently.
3. Use Dueling DQN to compare performance against Double DQN
