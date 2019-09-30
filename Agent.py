import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
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

        self.fc1 = nn.Linear(self.state_dim,256)
        self.policy = nn.Linear(256, self.action_dim)
        self.value = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)
        
    def get_action(self,x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.policy(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    
    def get_value(self,x):
        x = F.relu(self.fc1(x))
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
    
