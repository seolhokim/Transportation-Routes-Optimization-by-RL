import argparse
import numpy as np
import os
from environment.Building import Building
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = 'cpu'

#Hyperparameters

K_epoch       = 10
MAX_PASSENGERS_LENGTH = 40
MAX_ELV_LENGTH = 10

add_people_at_step = 25
add_people_prob = 0.8

print_interval = 20
global_step = 0


class Agent(nn.Module):
    def __init__(self, lift_num, building_height, max_people_in_floor, max_people_in_elevator,action_dim):
        super(Agent,self).__init__()
        self.memory = []
        self.maxpool = nn.MaxPool1d(2)
        '''
        floor,elv,elv_place
        floor shape : batch , 2 , (building_height * max_people_in_floor)
        elv shape : batch , lift_num , max_people_in_elevator
        elv_pace shape : batch , lift_num
        '''
        ##actor
        
        self.ac_floor = nn.Conv1d(2,32,3,padding=1)
        self.ac_elv = nn.Conv1d(lift_num,16,3,padding=1)
        self.ac_elv_place = nn.Linear(lift_num,16)
        
        self.ac_floor_1 = nn.Conv1d(32,32,3,padding=1)
        self.ac_elv_1 = nn.Conv1d(16,16,3,padding=1)
        
        self.ac_1 = nn.Linear(int(32*(building_height * max_people_in_floor)*(1/2)+16*max_people_in_elevator*(1/2)+16),360)
        self.ac_2 = nn.Linear(360,action_dim)
        
        
        ##value
        self.va_floor = nn.Conv1d(2,32,3,padding=1)
        self.va_elv = nn.Conv1d(lift_num,16,3,padding=1)
        self.va_elv_place = nn.Linear(lift_num,16)
        
        self.va_floor_1 = nn.Conv1d(32,32,3,padding=1)
        self.va_elv_1 = nn.Conv1d(16,16,3,padding=1)
        
        self.va_1 = nn.Linear(int(32*(building_height * max_people_in_floor)*(1/2)+16*max_people_in_elevator*(1/2)+16),360)
        self.va_2 = nn.Linear(360,action_dim)
        
        
        
        self.minibatch_size = 64
        self.gamma = 0.99
        self.lmbda = 0.95
        self.device = device
        self.lr_rate = 0.0003
        self.eps_clip = 0.2
        self.critic_coef = 0.5
        self.optimizer = optim.Adam(self.parameters(),lr = self.lr_rate)
    def get_action(self,floor,elv,elv_place,softmax_dim = -1):
        batch_size = floor.shape[0]
        
        floor = self.maxpool(torch.relu(self.ac_floor(floor)))
        floor = (torch.relu(self.ac_floor_1(floor)))
        elv = self.maxpool(torch.relu(self.ac_elv(elv)))
        elv = (torch.relu(self.ac_elv_1(elv)))
        elv_place = (torch.relu(self.ac_elv_place(elv_place)))
        
        floor = floor.view(batch_size,-1)
        elv = elv.view(batch_size,-1)
        
        x = torch.cat((floor,elv,elv_place),-1)
        x = self.ac_1(x)
        x = self.ac_2(torch.relu(x))
        
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    def get_value(self,floor,elv,elv_place):
        batch_size = floor.shape[0]
        
        floor = self.maxpool(torch.relu(self.va_floor(floor)))
        floor = (torch.relu(self.va_floor_1(floor)))
        elv = self.maxpool(torch.relu(self.va_elv(elv)))
        elv = (torch.relu(self.va_elv_1(elv)))
        elv_place = (torch.relu(self.va_elv_place(elv_place)))
        
        floor = floor.view(batch_size,-1)
        elv = elv.view(batch_size,-1)
        
        x = torch.cat((floor,elv,elv_place),-1)
        x = self.va_1(x)
        x = self.va_2(torch.relu(x))
        return x
    
    def put_data(self,data):
        self.memory.append(data)
        
    def make_batch(self):
        state_1_list,state_2_list,state_3_list, action_list, reward_list, next_state_1_list,next_state_2_list,next_state_3_list, prob_list, done_list = [],[],[],[],[],[],[],[],[],[]
        for data in self.memory:
            state_1,state_2,state_3,action,reward,next_state_1,next_state_2,next_state_3,prob,done = data
            state_1_list.append(state_1)
            state_2_list.append(state_2)
            state_3_list.append(state_3)
            action_list.append([action])
            reward_list.append([reward])
            prob_list.append([prob])
            next_state_1_list.append(next_state_1)
            next_state_2_list.append(next_state_2)
            next_state_3_list.append(next_state_3)
            done_mask = 0 if done else 1
            done_list.append([done_mask])
        self.memory = []
        
        s1,s2,s3,a,r,next_s1,next_s2,next_s3,done_mask,prob = torch.tensor(state_1_list,dtype=torch.float).to(device),\
                                            torch.tensor(state_2_list,dtype=torch.float).to(device),\
                                            torch.tensor(state_3_list,dtype=torch.float).to(device),\
                                        torch.tensor(action_list).to(device),torch.tensor(reward_list).to(device),\
                                        torch.tensor(next_state_1_list,dtype=torch.float).to(device),\
                                        torch.tensor(next_state_2_list,dtype=torch.float).to(device),\
                                        torch.tensor(next_state_3_list,dtype=torch.float).to(device),\
                                        torch.tensor(done_list,dtype = torch.float).to(device),\
                                        torch.tensor(prob_list).to(device)
        return s1.squeeze(1),s2.squeeze(1),s3.squeeze(1),a,r,next_s1.squeeze(1),next_s2.squeeze(1),next_s3.squeeze(1),done_mask,prob
    def choose_mini_batch(self, mini_batch_size, states1,states2,states3, actions, rewards, next_states1,next_states2,next_states3, done_mask, old_log_prob, advantages, returns,old_value):
        full_batch_size = len(states1)
        full_indices = np.arange(full_batch_size)
        np.random.shuffle(full_indices)
        for i in range(full_batch_size // mini_batch_size):
            indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
            yield states1[indices],states2[indices],states3[indices], actions[indices], rewards[indices], next_states1[indices],next_states2[indices],next_states3[indices], done_mask[indices],\
                  old_log_prob[indices], advantages[indices], returns[indices],old_value[indices]
    def train(self,summary,epoch):
        state_1_,state_2_,state_3_,action_,reward_, next_state_1_,next_state_2_,next_state_3_,done_mask_,action_prob_ = self.make_batch()
        old_value_ = self.get_value(state_1_,state_2_,state_3_).detach()
        td_target = reward_ + self.gamma * self.get_value(next_state_1_,next_state_2_,next_state_3_) * done_mask_
        delta = td_target - old_value_
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if done_mask_[idx] == 0:
                advantage = 0.0
            advantage = self.gamma * self.lmbda * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage_ = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        returns_ = advantage_ + old_value_
        advantage_ = (advantage_ - advantage_.mean())/(advantage_.std()+1e-3)
        for i in range(K_epoch):
            for state_1,state_2,state_3,action,reward,next_state_1,next_state_2,next_state_3,done_mask,action_prob,advantage,return_,old_value in self.choose_mini_batch(\
                                                                              self.minibatch_size ,state_1_,state_2_,state_3_, action_,reward_, next_state_1_,next_state_2_,next_state_3_, done_mask_, action_prob_,advantage_,returns_,old_value_): 
                #td_error = reward + self.gamma * self.get_value(next_state_1,next_state_2,next_state_3) * done_mask
                value = self.get_value(state_1,state_2,state_3).float()
                now_action = self.get_action(state_1,state_2,state_3,softmax_dim = -1)
                now_action = now_action.gather(1,action)

                ratio = torch.exp(torch.log(now_action) - torch.log(action_prob))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio , 1-self.eps_clip, 1 + self.eps_clip) * advantage
                actor_loss = - torch.min(surr1,surr2).mean()
                
                old_value_clipped = old_value + (value - old_value).clamp(-self.eps_clip,self.eps_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                
                critic_loss = 0.5 * self.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
                
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i == 0 :
                    summary.add_scalar('loss/actor_loss', actor_loss.item(), epoch)
                    summary.add_scalar('loss/critic_loss', critic_loss.item(), epoch)



