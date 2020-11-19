import argparse
import numpy as np
import os
from environment.Building import Building
import time

add_people_at_step = 25
add_people_prob = 0.8

print_interval = 20
global_step = 0

def main(): 
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 100)')
    parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate (default : 0.0001)')
    parser.add_argument('--lift_num', type=int, default=1, help='number of elevators ')
    parser.add_argument('--building_height', type=int, default=5, help='building height ')
    parser.add_argument('--max_people_in_floor', type=int, default=8, help='maximum people in one floor')
    parser.add_argument('--max_people_in_elevator', type=int, default=8, help='maximum people in one elevator')
    parser.add_argument("--load_file", type=str, default = 'no', help = 'load initial parameters')
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
    ave_reward = 0 
    
    for epoch in range(args.epochs):
        building.empty_building()
        while building.remain_passengers_num == 0 :
            building.generate_passengers(add_people_prob)
        state = building.get_state()
        done = False
        global_step = 0
        while not done:
            global_step += 1
            action = int(input("action"))
            reward = building.perform_action([action])
            next_state = building.get_state()
            state = next_state
            if args.test:
                os.system("cls")
                building.print_building(global_step)
                print(action)
                print('now reward : ',reward)
                time.sleep(1.5)
            if done or (global_step > 300):
                done = True
                break

        ave_reward += global_step 
        
        if epoch%args.print_interval==0 and epoch!=0:
            print("# of episode :{}, avg score : {:.1f}".format(epoch, ave_reward/args.print_interval))
            ave_reward = 0
        #if (epoch % args.save_interval == 0 )& (epoch != 0):
        #    torch.save(model.state_dict(), './model_weights/model_'+str(epoch))
if __name__ == '__main__':
    main()