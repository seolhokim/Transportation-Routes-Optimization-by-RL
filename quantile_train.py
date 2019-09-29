import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate = 0.0001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


class Agent(nn.Module):
    def __init__(self, state_dim,action_dim,quantile_num,learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantile_num = quantile_num
        
        super(Agent,self).__init__()
        self.memory = []

        self.fc1 = nn.Linear(self.state_dim,256)
        self.policy = nn.Linear(256, self.action_dim)
        self.value = nn.Linear(256, self.action_dim * self.quantile_num)
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)
        
    def get_action(self,x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.policy(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    
    def get_value(self,x):
        x = F.relu(self.fc1(x))
        x = self.value(x)
        x = x.view(-1,self.action_dim, self.quantile_num)
        return x
    
    def put_data(self,data):
        self.memory.append(data)
        
    def make_batch(self):
        state_list, action_list, reward_list, next_state_list, prob_list, done_list = [],[],[],[],[],[]
        for data in self.memory:
            state,action,reward,next_state,prob,done = data
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            prob_list.append([prob])
            next_state_list.append(next_state)
            done_mask = 0 if done else 1
            done_list.append([done_mask])
        self.memory = []
        
        s,a,r,next_s,done_mask,prob = torch.tensor(state_list,dtype=torch.float),\
                                        torch.tensor(action_list),torch.tensor(reward_list),\
                                        torch.tensor(next_state_list,dtype=torch.float),\
                                        torch.tensor(done_list,dtype = torch.float),\
                                        torch.tensor(prob_list)
        return s,a,r,next_s,done_mask,prob
    
    def train(self):

        state,action,reward, next_state,done_mask,action_prob = self.make_batch()
        
        for i in range(K_epoch):
            next_get_value = self.get_value(next_state)[np.arange(state.shape[0]),action.reshape(-1,)].detach()
            now_get_value = self.get_value(state)[np.arange(state.shape[0]),action.reshape(-1,)]

            diff = next_get_value.reshape(state.shape[0],self.quantile_num,1).\
                        repeat(1,1,self.quantile_num) \
                    - now_get_value.repeat(1,5).view(-1,5,5)
            
            value_loss = huber(diff) * (tau - (diff.detach()<0).float()).abs()
            value_loss = value_loss.mean(-1).mean(-1)
            
            td_error = reward + gamma * next_get_value.mean(-1) * done_mask
            
            delta = td_error - now_get_value.mean(-1)
            #delta = torch.mean(delta,1).reshape(-1,1)
            delta = delta.detach().numpy()
            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list,dtype = torch.float)
            
            
            now_action = self.get_action(state,softmax_dim = 1)
            now_action = now_action.gather(1,action)
            
            ratio = torch.exp(torch.log(now_action) - torch.log(action_prob))
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio , 1-eps_clip, 1 + eps_clip) * advantage
            
            
            loss = (-torch.min(surr1,surr2)).mean(-1)

            loss = (loss + 10*value_loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

import numpy as np
import os
from Building import Building
#from Agent import Agent
import time
#====================================================================================


#====================================================================================
#Building Setting
lift_num = 1
buliding_height = 5
max_people_in_floor = 8
max_people_in_elevator = 10

add_people_at_step = 25
add_people_prob = 0.8

#Create building with 4 elevators, height 10, max people 30 in each floor
building = Building(lift_num, buliding_height, max_people_in_floor,max_people_in_elevator)

#Agent controls each elevator
#agent = Agent(buliding_height, lift_num, 4)
#agent.reload(280)
#The goal is to bring down all the people in the building to the ground floor

epochs = 1000
max_steps = 100
global_step = 0
T_horizon = 20
reward_list = []
print_interval = 20

action_dim = 4
quantile_num = 5
model = Agent((buliding_height)+ max_people_in_elevator + (lift_num *2),action_dim,quantile_num,learning_rate)
model.load_state_dict(torch.load('./model_weights/quantile_model_300'))
print_interval = 20
ave_reward = 0 
tau = torch.Tensor((2 * np.arange(model.quantile_num) + 1) / (2.0 * model.quantile_num)).view(1, -1)


for epoch in range(epochs):
    building.empty_building()
    while building.target == 0 :
        building.generate_people(add_people_prob)
    state = building.get_state()
    done = False
    global_step = 0
    while not done:
        for t in range(T_horizon):
            global_step += 1
            if (global_step % 25 == 0) & global_step > 0 :
                #building.generate_people(add_people_prob/2)
                pass
            prev_people = building.get_arrived_people()
            action_prob = model.get_action(torch.from_numpy(np.array(state)).float())
            m = Categorical(action_prob)
            action = m.sample().item()
            building.perform_action([action])
            reward = building.get_reward(prev_people) 
            
            next_state = building.get_state()
            building.print_building(global_step)
            print(action)            
            time.sleep(1)
            finished = next_state.copy()
            del finished[5:7]
            if (sum(finished) == 0.0) :
                reward = 100
                done = True
            model.put_data((state, action, reward/10.0, next_state, action_prob[action].item(), done))
            state = next_state
            
            if done or global_step > 300:
                done = True
                break

        model.train()
    ave_reward += global_step 
    #print("Epoch: %d Step: %d Average Reward: %.4f"%(epoch, global_step, ave_reward/global_step))
    if epoch%print_interval==0 and epoch!=0:
        print("# of episode :{}, avg score : {:.1f}".format(epoch, ave_reward/print_interval))
        ave_reward = 0
    if (epoch % 100 == 0 )& (epoch != 0):
        pass

    reward_list.append(global_step)