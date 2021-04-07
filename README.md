# Optimize transportation routes by Reinforcement learning
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/seolhokim/elevator-reinforcementlearning-application)
- application of reinforcement learning to optimize transportation routes
                                  
<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/gif4.gif" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_1.PNG" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_2.PNG" width="250" height="200"></left>

## Description
  - At the beginning, passengers are created with their destination. Elevator knows where passengers want to go even in the other elevators.
  - Elevators should transport as many people as possible to their destinations as quickly as possible.
  - Reward is a negative value for the sum of people in a building and in elevators.
## Implemented Algorithms
  - PPO + GAE

## RUN

~~~
python main.py
~~~
  - if you want to change hyper-parameters, you can check "python main.py --help"
  - you just train and test basic model using main.py


## Thanks to...
  - [@shinseung428](https://github.com/shinseung428)
