import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from agent.agent import Agent ,is_finish
from environment.Building import Building

import argparse
import numpy as np
import os
import time

gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

add_people_at_step = 25
add_people_prob = 0.8
print_interval = 20
global_step = 0
MAX_PASSENGERS_LENGTH = 40
MAX_ELV_LENGTH = 10

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = 'cpu'

def main(): 
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs, (default: 100)')
    parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)')
    parser.add_argument('--lift_num', type=int, default=1, help='number of elevators ')
    parser.add_argument('--building_height', type=int, default=5, help='building height ')
    parser.add_argument('--max_people_in_floor', type=int, default=8, help='maximum people in one floor')
    parser.add_argument('--max_people_in_elevator', type=int, default=8, help='maximum people in one elevator')
    parser.add_argument("--load_file", type=str, default = 'no', help = 'load initial parameters')
    parser.add_argument("--save_interval", type=int, default = 500, help = 'save interval')
    parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval')
    args = parser.parse_args()
    print('args.test : ',args.test)
    print('args.epochs : ', args.epochs)
    print('args.lr_rate : ',args.lr_rate)
    print('args.lift_num : ', args.lift_num)
    print('args.building_height :',args.building_height)
    print('args.max_people_in_floor : ',args.max_people_in_floor )
    print('args.max_people_in_elevator :', args.max_people_in_elevator)
    print('args.load_file : ',args.load_file)
    print('args.save_interval :',args.save_interval)
    print('args.print_interval :',args.print_interval)

    building = Building(args.lift_num, args.building_height, args.max_people_in_floor,\
                        args.max_people_in_elevator)
    ave_steps = 0 
    model = Agent(4)
    
    summary = SummaryWriter()
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(args.epochs):
        building.empty_building()
        while building.remain_passengers_num == 0 :
            building.generate_passengers(add_people_prob)
        floor_state,elv_state,elv_place_state = building.get_state()
        floor_state = torch.tensor(floor_state).transpose(1,0).unsqueeze(0).float()
        floor_state = torch.cat((floor_state,-1* torch.ones((1,2,MAX_PASSENGERS_LENGTH - floor_state.shape[2]))),-1)/10.
        elv_state = torch.tensor(elv_state).unsqueeze(0).float()
        elv_state = torch.cat((elv_state,-1* torch.ones((1,1,MAX_ELV_LENGTH - elv_state.shape[2]))),-1)/10.
        elv_place_state = torch.tensor(elv_place_state).unsqueeze(0).float()/10.
        done = False
        global_step = 0
        while not done:
            global_step += 1
            action_prob = model.get_action(floor_state.to(device),elv_state.to(device),elv_place_state.to(device))[0]
            m = Categorical(action_prob)
            action = m.sample().item()
            reward = building.perform_action([action])
            next_floor_state,next_elv_state,next_elv_place_state = building.get_state()
            done = is_finish((next_floor_state,next_elv_state))
            next_floor_state = torch.tensor(next_floor_state).transpose(1,0).unsqueeze(0).float()
            next_floor_state = torch.cat((next_floor_state,-1* torch.ones((1,2,MAX_PASSENGERS_LENGTH - next_floor_state.shape[2]))),-1)/10.
            next_elv_state = torch.tensor(next_elv_state).unsqueeze(0).float()
            next_elv_state = torch.cat((next_elv_state,-1* torch.ones((1,1,MAX_ELV_LENGTH - next_elv_state.shape[2]))),-1)/10.
            next_elv_place_state = torch.tensor(next_elv_place_state).unsqueeze(0).float()/10.
            model.put_data((floor_state.tolist(),elv_state.tolist(),elv_place_state.tolist(), action, reward/100.0, next_floor_state.tolist(),next_elv_state.tolist(), next_elv_place_state.tolist(), action_prob[action].item(), done))
            floor_state = next_floor_state
            elv_state = next_elv_state
            elv_place_state = next_elv_place_state
            if args.test:
                os.system("cls")
                building.print_building(global_step)
                print(action)
                print('now reward : ',reward)
                
                time.sleep(1.5)
            if done or (global_step > 300):
                done = True
                break
        model.train(epoch)
        summary.add_scalar('reward', global_step, epoch)
        ave_steps += global_step 
        
        if epoch%args.print_interval==0 and epoch!=0:
            print("# of episode :{}, avg episodes : {:.1f}".format(epoch, ave_steps/args.print_interval))
            ave_steps = 0
        if (epoch % args.save_interval == 0 )& (epoch != 0):
            torch.save(model.state_dict(), './model_weights/model_'+str(epoch))
if __name__ == '__main__':
    main()