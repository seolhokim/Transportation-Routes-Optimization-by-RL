import torch	
import torch.nn as nn	
import torch.nn.functional as F	
import torch.optim as optim	
from torch.distributions import Categorical	

import numpy as np	
import os	
from Building import Building	
#from Agent import Agent	
import time	
#====================================================================================	


#====================================================================================	
#Building Setting	
lift_num = 2	
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

#Hyperparameters	
learning_rate = 0.0001	
gamma         = 0.99	
lmbda         = 0.95	
eps_clip      = 0.1	
K_epoch       = 3	
T_horizon     = 20	
class Agent(nn.Module):	
    def __init__(self, state_dim,elevator_num,action_dim,learning_rate):	
        self.state_dim = state_dim	
        self.elevator_num = elevator_num	
        self.action_dim = action_dim	

        super(Agent,self).__init__()	
        self.memory = []	

        self.fc1 = nn.Linear(self.state_dim,256)	
        self.fc2 = nn.Linear(256,256)	
        self.policy = nn.ModuleList([nn.Linear(256, self.action_dim) for _ in range(self.elevator_num)])	
        #self.policy = [nn.Linear(256, self.action_dim) for x in range(elevator_num)]	
        self.value = nn.Linear(256, 1)	
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)	

    def get_action(self,x, softmax_dim = 0):	
        x = F.relu(self.fc1(x))	
        x = F.relu(self.fc2(x))	
        xs = [layer(x) for layer in self.policy]	
        x = [F.softmax(x,dim = softmax_dim) for x in xs]	
        return x	


    def get_value(self,x):	
        x = F.relu(self.fc1(x))	
        x = F.relu(self.fc2(x))	
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

            now_action = torch.stack(now_action)	

            action_select = np.array([[x[0][idx] for x in action] for idx in range(self.elevator_num)])	
            action_select = torch.from_numpy(action_select).reshape(self.elevator_num,-1,1)	

            now_action = now_action.gather(2,action_select)	

            action_prob = action_prob.reshape(-1,2,1)	
            action_prob = action_prob.permute(1,0,2)	
            ratio = torch.exp(torch.log(now_action) - torch.log(action_prob))	


            surr1 = ratio * advantage	
            surr2 = torch.clamp(ratio , 1-eps_clip, 1 + eps_clip) * advantage	

            loss = - torch.min(surr1,surr2) + F.smooth_l1_loss(self.get_value(state),td_error.detach())	
            self.optimizer.zero_grad()	
            loss.mean().backward()	
            self.optimizer.step()	

model = Agent((buliding_height)+ (max_people_in_elevator +lift_num) * lift_num,2,4,learning_rate)	
print_interval = 20	
ave_reward = 0 	
model.load_state_dict(torch.load('./model_weights/multi_model_490000'))	

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
            action_prob = model.get_action(torch.from_numpy(np.array(state)).float())	
            m = [Categorical(x) for x in action_prob]	
            action = [x.sample().item() for x in m]	
            building.perform_action(action)	
            reward = building.get_reward() 	

            next_state = building.get_state()	
            os.system("cls")
            building.print_building(global_step)
            time.sleep(.7)
            finished = next_state.copy()	
            del finished[-4:]	
            if (sum(finished) == 0.0) :	
                reward = 100. #* building.target	
                done = True	
            #print(sum(finished))	
            #print('global_step : ',global_step,'state : ',state, 'action : ', action, 'reward : ',reward/float(first_state), 'done : ',done)	
            #print('global_step : ',global_step,'state : ',state, 'action : ', action, 'reward : ',reward/100., 'done : ',done)	
            state = next_state	
            building.print_building(global_step)	
            print(action)	
            time.sleep(1)	
            if done or (global_step > 300):	
                done = True	
                break	

    ave_reward += global_step 	
    #print("Epoch: %d Step: %d Average Reward: %.4f"%(epoch, global_step, ave_reward/global_step))	
    if epoch%print_interval==0 and epoch!=0:	
        print("# of episode :{}, avg score : {:.1f}".format(epoch, ave_reward/print_interval))	
        ave_reward = 0	
    reward_list.append(global_step) 