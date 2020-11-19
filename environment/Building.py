import numpy as np 
from environment.Elevator import Elevator
from environment.Passenger import Passenger

# Building Class
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

    def get_reward(self) -> float :
        '''
        make reward function to get better agent
        '''
        #res = self.get_arrived_people() - 1
        reward = self.get_arrived_passengers() - sum([len(x) for x in self.floors_information])\
                - sum([len(x.curr_passengers_in_elv) for x in self.elevators]) + self.cumulated_reward
        self.cumulated_reward = 0
        return reward
    def get_arrived_passengers(self) -> int :
        arrived_passengers=0
        for e in self.elevators:
            arrived_passengers += e.arrived_passengers_num
            e.arrived_passengers_num = 0
        return arrived_passengers

    def get_state(self) -> list:
        res = [float(len(elem))/float(self.max_passengers_in_floor) for idx, elem in enumerate(self.floors_information)]
        #엘리베이터에 탑승한 승객들의 목적지를 list형태로 res에 추가. 엘리베이터 갯수만큼 리스트형태로 추가됨.
        for e in self.elevators:
            temp_lst = []
            for p in e.curr_passengers_in_elv:
                temp_lst.append(p.return_dest())
            temp_lst.sort()
            [res.append((temp_lst[x] +1)/ self.max_floor) if x < len(temp_lst) else res.append(0.) for x in range(self.max_passengers_in_elevator) ]
        for e in self.elevators:
            res.append(float(e.curr_floor)/float(self.max_floor))
            res.append(float(len(e.curr_passengers_in_elv))/float(e.max_passengers))
        return res

    
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
                #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TODO
                self.floors_information[floor_num] += additional_passengers
                self.remain_passengers_num += passenger_num


    def perform_action(self, action):
        for idx,e in enumerate(self.elevators):
            if action[idx] == 3:
                '''
                elevator unloads passengers
                '''
                if len(e.curr_passengers_in_elv) == 0 :
                    self.cumulated_reward -= 1
                res = e.unload_passengers(self.floors_information[e.curr_floor])#, self.max_people_in_floor

            elif action[idx] == 2:
                '''
                elevator loads passengers
                '''
                if self.floors_information[e.curr_floor] == 0:
                    self.cumulated_reward -= 1
                self.floors_information[e.curr_floor] = e.load_passengers(self.floors_information[e.curr_floor])

            elif action[idx] == 1:
                '''
                elevator goes upstairs.
                '''
                if e.max_floor == e.curr_floor - 1:
                    self.cumulated_reward -= 1
                e.move_up()

            elif action[idx] == 0:
                '''
                elevator goes downstairs. 
                '''
                if e.curr_floor == 0 :
                    self.cumulated_reward -= 1                      
                e.move_down()

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
        print('now reward : ',self.get_reward())