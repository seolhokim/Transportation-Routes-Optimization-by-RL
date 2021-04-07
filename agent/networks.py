import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,building_height,max_people_in_floor,max_people_in_elevator,\
                          action_dim,lift_num):
        '''
        floor,elv,elv_place
        floor shape : batch , 2 , (building_height * max_people_in_floor)
        elv shape : batch , lift_num , max_people_in_elevator
        elv_pace shape : batch , lift_num
        '''
        super(Actor,self).__init__()
        self.action_dim = action_dim
        self.lift_num = lift_num
        self.ac_floor = nn.Conv1d(2,32,3,padding=1)
        self.ac_elv = nn.Conv1d(lift_num,16,3,padding=1)
        self.ac_elv_place = nn.Linear(lift_num,16)
        
        self.ac_floor_1 = nn.Conv1d(32,32,3,padding=1)
        self.ac_elv_1 = nn.Conv1d(16,16,3,padding=1)
        
        self.ac_1 = nn.Linear(int(32*(building_height * max_people_in_floor)*(1/4)+16*max_people_in_elevator*(1/4)+16),360)
        self.ac_2 = nn.Linear(360,action_dim*lift_num)
        
        self.maxpool = nn.MaxPool1d(2)
    def forward(self,floor,elv,elv_place,softmax_dim = -1):
        batch_size = floor.shape[0]
        
        floor = self.maxpool(torch.relu(self.ac_floor(floor)))
        floor = self.maxpool(torch.relu(self.ac_floor_1(floor)))
        elv = self.maxpool(torch.relu(self.ac_elv(elv)))
        elv = self.maxpool(torch.relu(self.ac_elv_1(elv)))
        elv_place = (torch.relu(self.ac_elv_place(elv_place)))
        
        floor = floor.view(batch_size,-1)
        elv = elv.view(batch_size,-1)
        
        x = torch.cat((floor,elv,elv_place),-1)
        x = self.ac_1(x)
        x = self.ac_2(torch.relu(x))
        
        x = x.view(-1,self.lift_num,self.action_dim)
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    
class Critic(nn.Module):
    def __init__(self,building_height,max_people_in_floor,max_people_in_elevator,\
                          action_dim,lift_num):
        '''
        floor,elv,elv_place
        floor shape : batch , 2 , (building_height * max_people_in_floor)
        elv shape : batch , lift_num , max_people_in_elevator
        elv_pace shape : batch , lift_num
        '''
        super(Critic,self).__init__()
        self.va_floor = nn.Conv1d(2,32,3,padding=1)
        self.va_elv = nn.Conv1d(lift_num,16,3,padding=1)
        self.va_elv_place = nn.Linear(lift_num,16)
        
        self.va_floor_1 = nn.Conv1d(32,32,3,padding=1)
        self.va_elv_1 = nn.Conv1d(16,16,3,padding=1)
        
        self.va_1 = nn.Linear(int(32*(building_height * max_people_in_floor)*(1/4)+16*max_people_in_elevator*(1/4)+16),360)
        self.va_2 = nn.Linear(360,1)
        
        self.maxpool = nn.MaxPool1d(2)
    def forward(self,floor,elv,elv_place):
        batch_size = floor.shape[0]
        
        floor = self.maxpool(torch.relu(self.va_floor(floor)))
        floor = self.maxpool(torch.relu(self.va_floor_1(floor)))
        elv = self.maxpool(torch.relu(self.va_elv(elv)))
        elv = self.maxpool(torch.relu(self.va_elv_1(elv)))
        elv_place = (torch.relu(self.va_elv_place(elv_place)))
        
        floor = floor.view(batch_size,-1)
        elv = elv.view(batch_size,-1)
        
        x = torch.cat((floor,elv,elv_place),-1)
        x = self.va_1(x)
        x = self.va_2(torch.relu(x))
        return x
    