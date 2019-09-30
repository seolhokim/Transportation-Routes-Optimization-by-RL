import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Agent import Agent
import numpy as np
import os
from Building import Building
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

def main(): #str
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)')
    parser.add_argument('--lift_num', type=int, default=1, help='number of elevators ')
    parser.add_argument('--building_height', type=int, default=5, help='building height ')
    parser.add_argument('--max_people_in_floor', type=int, default=8, help='maximum people in one floor')
    parser.add_argument('--max_people_in_elevator', type=int, default=8, help='maximum people in one elevator')
    parser.add_argument("--load_file", type=int, default = 0, help = 'load initial parameters')
    parser.add_argument("--save_interval", type=int, default = 1000, help = 'save interval')
    parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval')
    args = parser.parse_args()
    print('args.train : ',args.test)
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
    model = Agent((args.building_height)+ args.max_people_in_elevator + (args.lift_num *2),4,args.lr_rate)
    try:
        model.load_state_dict(torch.load("./model_weights/model_"+str(args.load_file)))
    except:
        pass
    ave_reward = 0 
    
    for epoch in range(args.epochs):
        building.empty_building()
        while building.target == 0 :
            building.generate_people(add_people_prob)
        state = building.get_state()
        done = False
        global_step = 0
        while not done:
            for t in range(T_horizon):
                global_step += 1
                
                action_prob = model.get_action(torch.from_numpy(np.array(state)).float())
                m = Categorical(action_prob)
                action = m.sample().item()
                building.perform_action([action])
                reward = building.get_reward() 

                next_state = building.get_state()
                finished = next_state.copy()
                del finished[-2:]
                if (sum(finished) == 0.0) :
                    reward = 100. 
                    done = True
                model.put_data((state, action, reward/100.0, next_state, action_prob[action].item(), done))
                state = next_state
                if args.test:
                    os.system("cls")
                    building.print_building(global_step)
                    print(action)
                    time.sleep(1.5)
                if done or (global_step > 300):
                    done = True
                    break

            model.train()
        ave_reward += global_step 
        
        if epoch%args.print_interval==0 and epoch!=0:
            print("# of episode :{}, avg score : {:.1f}".format(epoch, ave_reward/args.print_interval))
            ave_reward = 0
        if (epoch % args.save_interval == 0 )& (epoch != 0):
            torch.save(model.state_dict(), './model_weights/model_'+str(epoch))
if __name__ == '__main__':
    main()