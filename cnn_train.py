import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
#Hyperparameters
learning_rate = 0.0001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
class Agent(nn.Module):
    def __init__(self, state_dim,action_dim,learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        super(Agent,self).__init__()
        self.memory = []

        '''
        self.fc1 = nn.Linear(self.state_dim,256)
        self.policy = nn.Linear(256, self.action_dim)
        self.value = nn.Linear(256, 1)
        '''

        a = 1
        b = 1
        c = 4
        d = 256
        self.conv_layer_1_1 = nn.Conv1d(in_channels = 1,out_channels = a, kernel_size = 1, stride = 1,padding = 0,bias = False)
        self.conv_layer_1_2 = nn.Conv1d(in_channels = a,out_channels = a, kernel_size = 1, stride = 1,padding = 0,bias = False)
        
        self.conv_layer_2_1 = nn.Conv1d(in_channels = 1,out_channels = b, kernel_size = 1, stride = 1,padding = 0,bias = False)
        self.conv_layer_2_2 = nn.Conv1d(in_channels = b,out_channels = b, kernel_size = 1, stride = 1,padding = 0,bias = False)
        

        self.layer_1 = nn.Linear(2,c)
        
        self.result_layer = nn.Linear(a*5+b*10+c,d)
        self.result_layer_2 = nn.Linear(d,self.action_dim)
        self.value = nn.Linear(d,1)
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)

    def get_action(self,x, softmax_dim = 0):
        if softmax_dim == 1:
            floor_state = x.narrow(1,0,5) #.reshape(-1,5,1).permute(0,2,1)
            elevator_state = x.narrow(1,5,10) #.reshape(-1,10,1).permute(0,2,1)
            elevator_additional_state = x.narrow(1,15,2)
            
            floor_state = floor_state.reshape(floor_state.shape[0],1,-1)
            elevator_state = elevator_state.reshape(elevator_state.shape[0],1,-1)
        else:
            
            floor_state = x.narrow(0,0,5)
            elevator_state = x.narrow(0,5,10)
            elevator_additional_state = x.narrow(0,15,2)
            #print(x)
            #print(floor_state)
            #print(elevator_state)
            #print(elevator_additional_state)
            floor_state = floor_state.reshape(1,1,-1)
            elevator_state = elevator_state.reshape(1,1,-1)
        x_1 = F.relu(self.conv_layer_1_1(floor_state)) #F.relu
        x_1 = F.relu(self.conv_layer_1_2(x_1))
        x_1 = torch.flatten(x_1,start_dim=1)
        
        
        x_2 = F.relu(self.conv_layer_2_1(elevator_state))
        x_2 = F.relu(self.conv_layer_2_2(x_2))
        x_2 = torch.flatten(x_2,start_dim=1)

        x_3 = F.relu(self.layer_1(elevator_additional_state))
        #print('x_3.shape',x_3.shape)
        if len(x_3.shape) == 1:
            x_1 = torch.flatten(x_1,start_dim=0)
            x_2 = torch.flatten(x_2,start_dim=0)
            
        x = torch.cat((x_1,x_2,x_3),softmax_dim)
        #print('x_1.shape',x_1.shape)
        #print('x_2.shape',x_2.shape)
        #print('x_3.shape',x_3.shape)
        #print('x.shape',x.shape)
        #print(softmax_dim, " : ", x.shape)
        x = F.relu(self.result_layer(x))
        x = self.result_layer_2(x)
        
        x = F.softmax(x, dim = softmax_dim) # 하나만할때 0

        return x
    
    def get_value(self,x):
        floor_state = x.narrow(1,0,5) #.reshape(-1,5,1).permute(0,2,1)
        elevator_state = x.narrow(1,5,10) #.reshape(-1,10,1).permute(0,2,1)
        elevator_additional_state = x.narrow(1,15,2)
        
        floor_state = floor_state.reshape(floor_state.shape[0],1,-1)
        elevator_state = elevator_state.reshape(elevator_state.shape[0],1,-1)
        
        x_1 = F.relu(self.conv_layer_1_1(floor_state)) #F.relu
        x_1 = F.relu(self.conv_layer_1_2(x_1))
        x_1 = torch.flatten(x_1,start_dim=1)
        #print('x_1.shape',x_1.shape)
        
        x_2 = F.relu(self.conv_layer_2_1(elevator_state))
        x_2 = F.relu(self.conv_layer_2_2(x_2))
        x_2 = torch.flatten(x_2,start_dim=1)

        x_3 = F.relu(self.layer_1(elevator_additional_state))

        x = torch.cat((x_1,x_2,x_3), 1)
        
        x = F.relu(self.result_layer(x))
        x = self.value(x)
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
            td_error = reward + gamma * self.get_value(next_state) * done_mask
            delta = td_error - self.get_value(state)
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
            loss = - torch.min(surr1,surr2) + F.smooth_l1_loss(self.get_value(state),td_error.detach())
            
            self.optimizer.zero_grad()
            loss.mean().backward()
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


model = Agent((buliding_height)+ max_people_in_elevator + (lift_num *2),4,learning_rate)
model.load_state_dict(torch.load('./model_weights/cnn_model_100'))
print_interval = 20
ave_reward = 0 




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
            finished = next_state.copy()
            del finished[-2]
            #print(action)
            #print('next_state',next_state)
            #print('finished',finished)
            if (sum(finished) == 0.0) :
                reward = 100.
                done = True
                
            model.put_data((state, action, reward/10., next_state, action_prob[action].item(), done))
            state = next_state
            building.print_building(global_step)
            print(action)
            time.sleep(1)
            if done or global_step > 300:
                done = True
                break
        #state,action,reward, next_state,done_mask,action_prob = model.make_batch()
        #raise Exception()
        model.train()
        #raise Exception()
    ave_reward += global_step 
    #print("Epoch: %d Step: %d Average Reward: %.4f"%(epoch, global_step, ave_reward/global_step))
    if epoch%print_interval==0 and epoch!=0:
        print("# of episode :{}, avg score : {:.1f}".format(epoch, ave_reward/print_interval))
        ave_reward = 0
    if (epoch % 100 == 0 )& (epoch != 0):
        torch.save(model.state_dict(), './model_weights/cnn_model_'+str(epoch))
    reward_list.append(global_step)