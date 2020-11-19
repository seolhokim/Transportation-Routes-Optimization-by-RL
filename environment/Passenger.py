import random

class Passenger(object):
    '''
    The main concern of this problem is passenger transportation.
    The number of passenger is randomly generated on all floor.
    And destination of all passenger us also randomly generated.
    '''
    def __init__(self,now_floor,max_floor):
        self.now_floor = now_floor
        self.dest = random.choice(list(range(now_floor)) + list(range(now_floor+1,max_floor)))
    def return_dest(self):
        return self.dest


