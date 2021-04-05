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
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
learning_rate = 0.001
MAX_PASSENGERS_LENGTH = 40
MAX_ELV_LENGTH = 10

add_people_at_step = 25
add_people_prob = 0.8

print_interval = 20
global_step = 0

class SequenceEncoder(nn.Module):
    def __init__(self,input_dim,seq_len,encoding_dim = 128,head_num = 16, normalization=True):
        super(SequenceEncoder,self).__init__()
        self.normalization = normalization
        self.linear_1 = nn.Linear(input_dim,encoding_dim)
        self.attention = nn.MultiheadAttention(encoding_dim, head_num)
        
        self.conv_1 = nn.Conv1d(encoding_dim,encoding_dim,1)
        self.conv_2 = nn.Conv1d(encoding_dim,encoding_dim*1,1)
        self.norm = nn.BatchNorm1d(encoding_dim*1)
        ##self.lstm = nn.LSTM(input_size = encoding_dim,hidden_size = encoding_dim,num_layers = 1,bidirectional=True,batch_first = True)
        ##self.linear_2 = nn.Linear(encoding_dim*2,encoding_dim)
        self.linear_2 = nn.Linear(seq_len,1)
    def forward(self,x):
        '''
        #(1,3,40)
        (batch_size, hidden_size, max_seq_len)
        '''
        x = x.permute(0,2,1) #(batch_size, max_seq_len, hidden_size)
        x = self.linear_1(x) #(batch_size, max_seq_len, encoding_dim)
        x = x.permute(1,0,2)#(max_seq_len, batch_size, encoding_dim)
        x,_ = self.attention(x,x,x) #(max_seq_len, batch_size, encoding_dim)
        x = x.permute(1,2,0) #(batch_size, encoding_dim,max_seq_len)
        
        x_ = F.relu(self.conv_1(x)) #(batch_size, encoding_dim,max_seq_len)
        x_ = self.conv_2(x_) #(batch_size, encoding_dim,max_seq_len)
        
        x = x_ + x
        ####if self.normalization == True:
        ####    x = self.norm(x)
        #x(batch_size, encoding_dim,max_seq_len)
        ##x,_ = self.lstm(F.relu(x.transpose(2,1)))
        ##x = x[:,-1,:]
        ##x = self.linear_2(F.relu(x))
        x = self.linear_2(F.relu(x)).squeeze(-1)
        return x
class Agent(nn.Module):
    def __init__(self, action_dim):
        super(Agent,self).__init__()
        self.memory = []
        self.floor_encoder = SequenceEncoder(2,40)
        self.elv_encoder = SequenceEncoder(1,10)
        self.elv_place_encoder_1 = nn.Linear(1,32)
        self.elv_place_encoder_2 = nn.Linear(32,64)
        self.action_1 = nn.Linear(256+64,128)
        self.action_2 = nn.Linear(128,action_dim)
        self.value_1 = nn.Linear(256+64,128)
        self.value_2 = nn.Linear(128,1)
        
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)
        
    def get_action(self,floor,elv,elv_place,softmax_dim = -1):
        #(batch_size, hidden_size, max_seq_len)
        floor = self.floor_encoder(floor)
        elv = self.elv_encoder(elv)
        elv_place = F.relu(self.elv_place_encoder_1(elv_place))
        elv_place = (self.elv_place_encoder_2(elv_place))
        x = torch.cat((floor,elv,elv_place),-1)
        #x = F.relu(x)
        action = F.relu(self.action_1(x))
        action = self.action_2(action)
        prob = F.softmax(action, dim = softmax_dim)
        return prob
    def get_value(self,floor,elv,elv_place):
        floor = self.floor_encoder(floor)
        elv = self.elv_encoder(elv)
        elv_place = F.relu(self.elv_place_encoder_1(elv_place))
        elv_place = (self.elv_place_encoder_2(elv_place))
        x = torch.cat((floor,elv,elv_place),-1)
        #x = F.relu(x)
        
        value = self.value_1(x)
        value = self.value_2(value)
        return value
    
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
    
    def train(self,summary,epoch):
        state_1,state_2,state_3,action,reward, next_state_1,next_state_2,next_state_3,done_mask,action_prob = self.make_batch()

        for i in range(K_epoch):
            td_error = reward + gamma * self.get_value(next_state_1,next_state_2,next_state_3) * done_mask
            delta = td_error - self.get_value(state_1,state_2,state_3)
            if torch.cuda.is_available():
                delta = delta.cpu().detach().numpy()
            else:
                delta = delta.detach().numpy()
            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list,dtype = torch.float).to(device)
            
            
            now_action = self.get_action(state_1,state_2,state_3,softmax_dim = -1)
            now_action = now_action.gather(1,action)
            
            ratio = torch.exp(torch.log(now_action) - torch.log(action_prob))
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio , 1-eps_clip, 1 + eps_clip) * advantage
            loss_1 = - torch.min(surr1,surr2).mean()
            loss_2 = F.smooth_l1_loss(self.get_value(state_1,state_2,state_3),td_error.detach())
            loss = loss_1 + loss_2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i == 0 :
                summary.add_scalar('loss/loss_1', loss_1.item(), epoch)
                summary.add_scalar('loss/loss_2', loss_2.item(), epoch)



