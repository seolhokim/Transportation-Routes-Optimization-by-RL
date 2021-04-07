import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils import Rollouts
from agent.networks import Actor,Critic
class Agent(nn.Module):
    def __init__(self,device, lift_num, building_height, max_people_in_floor, max_people_in_elevator,action_dim,K_epoch,gamma,lmbda,lr_rate,eps_clip,critic_coef,minibatch_size):
        super(Agent,self).__init__()
        self.data = Rollouts()
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.lr_rate = lr_rate
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.critic_coef = critic_coef
        
        self.minibatch_size = minibatch_size
        self.actor = Actor(building_height,max_people_in_floor,max_people_in_elevator,\
                          action_dim,lift_num)
        self.critic = Critic(building_height,max_people_in_floor,max_people_in_elevator,\
                          action_dim,lift_num)

        self.optimizer = optim.Adam(self.parameters(),lr = self.lr_rate)
    def get_action(self,floor,elv,elv_place,softmax_dim = -1):
        return self.actor(floor,elv,elv_place,softmax_dim)
        
    def get_value(self,floor,elv,elv_place):
        return self.critic(floor,elv,elv_place)
    
    def put_data(self,transition):
        self.data.append(transition)
        
    def train(self,summary,epoch):
        state_1_,state_2_,state_3_,action_,reward_, next_state_1_,next_state_2_,next_state_3_,done_mask_,action_prob_ = self.data.make_batch(self.device)
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
        for i in range(self.K_epoch):
            for state_1,state_2,state_3,action,reward,next_state_1,next_state_2,next_state_3,done_mask,action_prob,advantage,return_,old_value in self.data.choose_mini_batch(\
                                                                              self.minibatch_size ,state_1_,state_2_,state_3_, action_,reward_, next_state_1_,next_state_2_,next_state_3_, done_mask_, action_prob_,advantage_,returns_,old_value_): 
                value = self.get_value(state_1,state_2,state_3).float()
                now_action = self.get_action(state_1,state_2,state_3,softmax_dim = -1)
                now_action = now_action.gather(2,action.unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp((torch.log(now_action) - torch.log(action_prob)).sum(-1,keepdim=True))

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

