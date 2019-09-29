import numpy as np 
from Elevator import Elevator
from Passenger import Passenger

class Switch(object):
	def __init__(self):
		self.up = " "
		self.down = " "

	def reset(self):
		self.up = " "
		self.down = " "

# Building Class
class Building(object):
	def __init__(self, total_elevator_num, height, max_people,max_people_in_elevator):
		self.target = 0
		self.cumulated_reward = 0
		self.total_elevator_num = total_elevator_num
		self.max_people_in_floor = max_people
		self.max_people_in_elevator = max_people_in_elevator
		#each elevator has max capacity of 10
		self.elevators = []
		for idx in range(total_elevator_num):
			self.elevators.append(Elevator(idx, 10, height))

		self.height = height
		self.people_in_floors = []
		self.floor_button = []
		for idx in range(height):
			self.people_in_floors.append([])
			self.floor_button.append(Switch())

	def get_reward(self):
		#res = self.get_arrived_people() - 1
		res = self.get_arrived_people() - sum([len(x) for x in self.people_in_floors])\
				- sum([len(x.curr_people) for x in self.elevators]) + self.cumulated_reward
		#res = - sum([len(x) for x in self.people_in_floors[1:]])\
		#		- sum([len(x.curr_people) for x in self.elevators])\
		#		+ self.get_arrived_people() + (self.cumulated_reward *10)
		self.cumulated_reward = 0
		#print(res)
		return res
	#self.cumulated_reward
	# check number of people in ground floor

	def get_arrived_people(self):
		# 이 층에서 내리는 인원 리턴.
		off_people=0
		for e in self.elevators:
			off_people += e.off_people
			#print(e.off_people)
			e.off_people=0
		return off_people

	# this function is not currently used
	def get_wait_time(self):
		total = 0
		for people in self.people_in_floors[1:]:
			for p in people:
				total += p.wait_time

		for elevator in self.elevators:
			for p in elevator.curr_people:
				total += p.wait_time
		return total


	# state of the building will be fed into the network as an input
	def get_state(self):
		#print('self.max_people_in_floor : ',self.max_people_in_floor)
		#print('self.target',self.target)

            
		#res = [float(len(elem))/float(self.max_people_in_floor) if idx > 0 else float(len(elem))/float(self.target) for idx, elem in enumerate(self.people_in_floors)]
		res = [float(len(elem))/float(self.max_people_in_floor) for idx, elem in enumerate(self.people_in_floors)]
		#엘리베이터에 탑승한 승객들의 목적지를 list형태로 res에 추가. 엘리베이터 갯수만큼 리스트형태로 추가됨.
		for e in self.elevators:
			temp_lst = []
			for p in e.curr_people:
				temp_lst.append(p.return_dest())
			temp_lst.sort()
			[res.append((temp_lst[x] +1)/ self.height) if x < len(temp_lst) else res.append(0.) for x in range(self.max_people_in_elevator) ]
		for e in self.elevators:
			res.append(float(e.curr_floor)/float(self.height))
			res.append(float(len(e.curr_people))/float(e.max_people))
		return res

	# clears the building 
	def empty_building(self):
		self.people_in_floors = []
		for idx in range(self.height):
			self.people_in_floors.append([])

		for e in self.elevators:
			e.empty()
		self.target = 0

	def generate_people(self, prob):
		#generate random people in building and button press in each floor
		for floor_num in range(0, self.height):
			if np.random.random() < prob and len(self.people_in_floors[floor_num]) < self.max_people_in_floor:
				people = np.random.randint(1,6)
				if len(self.people_in_floors[floor_num]) + people > self.max_people_in_floor:
					people = self.max_people_in_floor - (len(self.people_in_floors[floor_num]) + people)

				tmp_list = []
				for p in range(people):
					tmp_list.append(Passenger(nowFloor=floor_num, maxHeight=self.height))
				#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TODO
				self.people_in_floors[floor_num] += tmp_list
				self.target += people

 				# if np.random.random() < 0.5 and floor_num < self.height:
				# 	self.floor_button[floor_num].up = "^"
				# elif floor_num > 0:
				# 	self.floor_button[floor_num].down = "v"
		

	# actions can be redefined
	def perform_action(self, action):
		for idx,e in enumerate(self.elevators):
			if action[idx] == 3:
				# print "unload"
				if len(e.curr_people) == 0 :
					self.cumulated_reward -= 1
				res = e.unload_people(self.people_in_floors[e.curr_floor], self.max_people_in_floor)
				#for p in res:
				#	self.people_in_floors[e.curr_floor].append(p)

			elif action[idx] == 2:
				# print "load"
				if self.people_in_floors[e.curr_floor] == 0:
					self.cumulated_reward -= 1
				self.people_in_floors[e.curr_floor] = e.load_people(self.people_in_floors[e.curr_floor])
			
			elif action[idx] == 1:
				# print "up"
				if e.max_height == e.curr_floor - 1:
					self.cumulated_reward -= 1
				e.move_up()

			elif action[idx] == 0:
				# print "down"
				if e.curr_floor == 0 :
					self.cumulated_reward -= 1                      
				e.move_down()
	def increment_wait_time(self):
		for people in self.people_in_floors[1:]:
			for p in people:
				p.wait_time+=1

		for elevator in self.elevators:
			for p in elevator.curr_people:
				p.wait_time+=1

	def print_building(self, step):
		for idx in reversed(list(range(1,self.height))):
			print("=======================================================")
			print("= Floor #%02d ="%idx, end=' ')
			for e in self.elevators:
				if e.curr_floor == idx:
					print("  Lift #%d"%e.idx, end=' ')
				else:
					print("         ", end=' ')

			print(" ")
			# print "=   %c  %c   ="%(self.floor_button[idx].up, self.floor_button[idx].down),
			print("=  Waiting  =", end=' ')
			for e in self.elevators:
				if e.curr_floor == idx:
					print("    %02d   "%len(e.curr_people), end=' ')
				else:
					print("          ", end=' ')
			print(" ")
			print("=    %03d    ="%len(self.people_in_floors[idx]))


		print("=======================================================")
		print("= Floor #00 =", end=' ')
		for e in self.elevators:
			if e.curr_floor == 0:
				print("  Lift #%d"%e.idx, end=' ')
			else:
				print("         ", end=' ')

		print(" ")
		# print "=   %c  %c   ="%(self.floor_button[idx].up, self.floor_button[idx].down),
		print("=  Arrived  =", end=' ')
		for e in self.elevators:
			if e.curr_floor == 0:
				print("    %02d   "%len(e.curr_people), end=' ')
			else:
				print("          ", end=' ')		
		print(" ")
		print("=    %03d    ="%len(self.people_in_floors[0]))
		print("=======================================================")
		print("")
		print("People to move: %d "%(self.target - len(self.people_in_floors[0])))
		print("Total # of people: %d"%self.target)
		print("Step: %d"%step)
		print('state : ',self.get_state())
		print('now reward : ',self.get_reward())