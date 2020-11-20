import numpy as np 

class Elevator(object):
    '''
    Elevator can move passenger to another floor.
    If elevator transports passenger to his(her) destination well, then it occurs positive reward.
    The elevator can only carry a limited number of people at a time.
    '''
    def __init__(self, idx : int, max_passengers : int, max_floor : int):
        '''
        idx(int) : elevator name to classify
        max_passengers(int) : limited number of passengers at a time.
        max_floor(int) : maximum number of floors in a building
        
        curr_floor(int) : elevator's current floor 
        curr_passengers_in_elv(list(Passanger)) : the passengers who are transported by this elevator.
        arrived_passengers_num(int) : number of arrived passengers for calculate reward
        '''
        self.idx = idx
        self.max_floor = max_floor
        self.max_passengers = max_passengers
        self.curr_floor = 0
        self.curr_passengers_in_elv = []
        self.arrived_passengers_num = 0 

    def move_up(self):
        if self.curr_floor < self.max_floor-1:
            self.curr_floor += 1

    def move_down(self):
        if self.curr_floor > 0:
            self.curr_floor -= 1

    def empty(self):
        self.curr_passengers_in_elv = []
        self.curr_floor = 0

    def load_passengers(self, passengers_in_floor : int) -> list:
        '''
        function loads passengers into the elevator
        res(list(Passenger)) : passengers who are unable to get on elevator because of the elevator's maximum capacity.
        '''
        if len(passengers_in_floor) > (self.max_passengers - len(self.curr_passengers_in_elv)):
            #FIFO
            res = passengers_in_floor[self.max_passengers - len(self.curr_passengers_in_elv):]
            for p in passengers_in_floor[:self.max_passengers - len(self.curr_passengers_in_elv)]:
                self.curr_passengers_in_elv.append(p)
        else:
            for p in passengers_in_floor:
                self.curr_passengers_in_elv.append(p)
            res = []
        return res
    def get_passengers_info(self) -> list :
        return [p.get_dest() for p in self.curr_passengers_in_elv]
    def unload_passengers(self, passengers_in_floor : list) -> int:
        '''
        function unloads passengers back into the building 
        
        arrived_passengers(list(Passenger)) : list of passengers to drop off on this floor
        '''
        arrived_passengers = []
        num_in_floor = len(passengers_in_floor)
        self.arrived_passengers_num = 0
        for i in range(len(self.curr_passengers_in_elv)):
            if self.curr_passengers_in_elv[i].dest == self.curr_floor:
                arrived_passengers.append(i)
        #If anyone gets off this floor
        if len(arrived_passengers) !=0:
            self.arrived_passengers_num = len(arrived_passengers)
            arrived_passengers.reverse() #TODO : Debugging
            for i in arrived_passengers:
                self.curr_passengers_in_elv.pop(i)

        '''
        for p in ((self.curr_passengers_in_elv)):
            if p.dest == self.curr_floor:
                self.curr_passengers_in_elv.remove(p)
                self.arrived_passengers_num += 1
        '''
        return self.arrived_passengers_num