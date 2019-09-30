# Elevator Computation Research
- application of reinforcement learning to improve elevator performance

<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_1.PNG" width="250" height="200"></left>
<left><img src="https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/env_2.PNG" width="250" height="200"></left>

## Experiments

| Elevator Numbers | Building Heights | Max People in Floor| Max People in Elevator |Epochs | mean convergence point |
| :---: | :---: | :---: | :---: |
| 1 | 5 | 8 | 8 | 15,168 | 18.6
| 1 | 6 | 6 | 6 | 100,000 | 23.6
| 1 | 7 | 7 | 7 | 100,000 | 29.7
| 1 | 8 | 8 | 5 | 100,000 | 35.3
| 2 | 5 | 8 | 8 | 500,000 | 14.5


## Todo
  - Implement Hierarchy PPO
  - More Test!
  - Clean up the directory

## Implemented Algorithms
  - PPO + GAE
    * one agent
    * multi agent
    * multi environment (more robust)
  - Distributional PPO(QR PPO)
    * one agent

## Update News


## Convergance Point Image

### liftnum : 1, buildingheight : 3, maxpeopleinfloor : 3, maxpeopleinelevator : 3
![img](https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/converge_point.PNG)

### Computation Increasing according to the Elevator's increase
![img](https://github.com/seolhokim/ppo_pytorch_elevator/blob/master/assets/computation_increase.PNG)

