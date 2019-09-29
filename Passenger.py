import random

#Passenger Class
class Passenger(object):
	def __init__(self,nowFloor,maxHeight):
		self.wait_time =0
		if nowFloor==0:
			self.dest = random.randint(1,maxHeight-1)
		else:
			self.dest = 0


	def return_wait_time(self):
		return self.wait_time

	def return_dest(self):
		return self.dest


