# Elevator Computation Research
- application of reinforcement learning to improve elevator performance

<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_1.PNG" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_2.PNG" width="250" height="200"></left>

## Experiments

| Elevator Numbers | Building Heights | Max People in Floor| Max People in Elevator | Epochs | mean convergence point |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5 | 8 | 8 | 15,168 | 18.6
| 1 | 6 | 6 | 6 | 100,000 | 23.6
| 1 | 7 | 7 | 7 | 100,000 | 29.7
| 1 | 8 | 8 | 5 | 100,000 | 35.3
| 2 | 5 | 8 | 8 | 500,000 | 14.5


## Todo
  - Implement Hierarchy PPO
  - More Test!
  - More insight
  - Clean up the directory

## Implemented Algorithms
  - PPO + GAE
    * one agent
    * multi agent
    * multi environment (more robust)
  - Distributional PPO(QR PPO)
    * one agent

## Update News

### 2019.09.03
  - Readme is updated
  - directory is cleaned up.

## Convergance Point Image

### liftnum : 1, buildingheight : 3, maxpeopleinfloor : 3, maxpeopleinelevator : 3
![img](https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/converge_point.PNG)

### Computation Increasing according to the Elevator's increase
![img](https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/computation_increase.PNG)
  - why Variance is increased?
  * -> Cause States' dimension is increased.

### Distributional Critic Network test
![img](https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/distributional.PNG)
  - i have thought that distributional critic network can enhance the performance in pomdp. cause model doesn't know passengers' destiny, So distributional reward mapping will be helpful! but it doesn't
  - i tried almost random space!
### Reference
  -Trust Region Policy Optimization
  -Proximal Policy Optimization Algorithm
  -A Distributional Perspective on Reinforcement Learning
  -Distributional Reinforcement Learning with Quntile Regression
  -Deep Hierarchical Reinforcement Learning Algorithm in Partially Observable Markov Decision Processes
  -Deterministic Policy Gradient Algorithm
  -Continuous Control with Deep Reinforcement Learning 
  -A Natural Policy Gradient
  -Asynchronous Methods for Deep Reinforcement Learning 
  -High-Demensional Continuous control using Generalized Advantage Estimation
  -Deep Recurrent Q-Learning for Partially Observable MDPs
  -Hierarchical Deep Reinforcement Learning : Integrating Temporal Abstraction and Intrinsic Motivation
  -Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

### Thanks to...
  - [@shinseung428](https://github.com/shinseung428)
  - [@seungeunrho](https://github.com/seungeunrho)
