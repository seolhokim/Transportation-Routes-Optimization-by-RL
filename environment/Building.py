import numpy as np

from environment.Elevator import Elevator
from environment.Passenger import Passenger


class Building(object):
    '''
    Building controls elevators and passengers.
    It sets constraints and operates entire environment.
    '''
    def __init__(self, total_elevator_num : int, max_floor : int, max_passengers_in_floor : int,max_passengers_in_elevator : int, elevator_capacity : int = 10):
        '''
        remain_passengers_num(int) : remain passengers in building
        total_elevator_num(int) : total number of elevator
        max_passengers_in_floor(int) : maximum number of passengers in one floor
        max_passengers_in_elevator(int) : maximum number of one elevator
        max_floor(int) : maximum floor in the building
        floors_information(list(Passenger)) : passenger's information on each floor
        elevators(list(Elevator)) : elevator list
        '''
        self.remain_passengers_num = 0
        self.cumulated_reward = 0
        self.total_elevator_num = total_elevator_num
        self.max_passengers_in_floor = max_passengers_in_floor
        self.max_passengers_in_elevator = max_passengers_in_elevator
        self.elevators = []
        for idx in range(total_elevator_num):
            self.elevators.append(Elevator(idx, elevator_capacity, max_floor))
        self.max_floor = max_floor
        self.floors_information = []
        for idx in range(max_floor):
            self.floors_information.append([])
    
    def get_arrived_passengers(self) -> int :
        arrived_passengers=0
        for e in self.elevators:
            arrived_passengers += e.arrived_passengers_num
            e.arrived_passengers_num = 0
        return arrived_passengers

    def get_state(self) -> tuple:
        floor_passengers = [[[floor,passenger.get_dest()] for passenger in passengers] for floor, passengers in enumerate(self.floors_information)]
        floor_passengers = [x for x in floor_passengers if x != []]
        floor_passengers = [y for x in floor_passengers for y in x]
        if len(floor_passengers) == 0 :
            floor_passengers.append([-1,-1])
        elv_passengers = [e.get_passengers_info() for e in self.elevators]
        elv_passengers = [x for x in elv_passengers if x != []]
        if len(elv_passengers) == 0 :
            elv_passengers.append([-1])
        elevators_floors = [e.curr_floor for e in self.elevators]
        return floor_passengers,elv_passengers,elevators_floors
    
    def empty_building(self):
        '''
        clears the building 
        '''
        self.floors_information = []
        
        for idx in range(self.max_floor):
            self.floors_information.append([])
        for e in self.elevators:
            e.empty()
        self.remain_passengers_num = 0
    def get_remain_passengers_in_building(self):
        return sum([len(x) for x in self.floors_information]) 
    def get_remain_passengers_in_elv(self,elv):
        return len(elv.curr_passengers_in_elv)
    def get_remain_all_passengers(self):
        return sum([self.get_remain_passengers_in_elv(x) for x in self.elevators]) +  self.get_remain_passengers_in_building()
    def generate_passengers(self, prob : float, passenger_max_num : int = 6):
        '''
        generate random people in building and button press in each floor
        '''
        for floor_num in range(0, self.max_floor):
            if np.random.random() < prob and len(self.floors_information[floor_num]) < self.max_passengers_in_floor:
                passenger_num = np.random.randint(1,passenger_max_num)
                passenger_num = min(self.max_passengers_in_floor, len(self.floors_information[floor_num]) + passenger_num)
                additional_passengers = []
                for p in range(passenger_num):
                    additional_passengers.append(Passenger(now_floor=floor_num, max_floor=self.max_floor))
                self.floors_information[floor_num] += additional_passengers
                self.remain_passengers_num += passenger_num

    def perform_action(self, action : list):
        arrived_passengers_num_lst = []
        penalty_lst = []
        for idx,e in enumerate(self.elevators):
            if action[idx] == 0 :
                '''
                elevator goes downstairs. 
                '''
                if e.curr_floor == 0 :
                    penalty_lst.append(-1)
                e.move_down()
            elif action[idx] == 1:
                '''
                elevator goes upstairs.
                '''
                if (e.max_floor-1) == (e.curr_floor):
                    penalty_lst.append(-1)
                e.move_up()
            elif action[idx] == 2:
                '''
                elevator loads passengers
                '''
                if len(self.floors_information[e.curr_floor]) == 0:
                    penalty_lst.append(-1)
                self.floors_information[e.curr_floor] = e.load_passengers(self.floors_information[e.curr_floor])
            elif action[idx] == 3:
                '''
                elevator unloads passengers
                '''
                arrived_passengers_num = e.unload_passengers(self.floors_information[e.curr_floor])
                if arrived_passengers_num == 0 :
                    penalty_lst.append(-1)
                arrived_passengers_num_lst.append(arrived_passengers_num)

        reward = sum(arrived_passengers_num_lst) + sum(penalty_lst) - self.get_remain_all_passengers()
        return reward
        #return self.get_reward()
    
    def print_building(self, step : int):
        for idx in reversed(list(range(1,self.max_floor))):
            print("=======================================================")
            print("= Floor #%02d ="%idx, end=' ')
            for e in self.elevators:
                if e.curr_floor == idx:
                    print("  Lift #%d"%e.idx, end=' ')
                else:
                    print("         ", end=' ')
            print(" ")
            print("=  Waiting  =", end=' ')
            for e in self.elevators:
                if e.curr_floor == idx:
                    print("    %02d   "%len(e.curr_passengers_in_elv), end=' ')
                else:
                    print("          ", end=' ')
            print(" ")
            print("=    %03d    ="%len(self.floors_information[idx]))
        print("=======================================================")
        print("= Floor #00 =", end=' ')
        for e in self.elevators:
            if e.curr_floor == 0:
                print("  Lift #%d"%e.idx, end=' ')
            else:
                print("         ", end=' ')
        print(" ")
        print("=  Arrived  =", end=' ')
        for e in self.elevators:
            if e.curr_floor == 0:
                print("    %02d   "%len(e.curr_passengers_in_elv), end=' ')
            else:
                print("          ", end=' ')		
        print(" ")
        print("=    %03d    ="%len(self.floors_information[0]))
        print("=======================================================")
        print("")
        print("People to move: %d "%(self.remain_passengers_num - len(self.floors_information[0])))
        print("Total # of people: %d"%self.remain_passengers_num)
        print("Step: %d"%step)
        print('state : ',self.get_state())
        #print('now reward : ',self.get_reward())