import numpy as np 


class Elevator(object):
	def __init__(self, num, max_people, max_height):
		self.idx = num
		self.max_height = max_height
		self.max_people = max_people
		#we don't need this right now since everyone's going down to the ground floor 
		# self.dest_floors = []
		self.curr_floor = 0
		self.curr_people = []

		# 그 순간에 내릴 인원. Building의 get_arrived_people에서 사용할거임
		self.off_people=0

	def move_up(self):
		if self.curr_floor < self.max_height-1:
			self.curr_floor += 1

	def move_down(self):
		if self.curr_floor > 0:
			self.curr_floor -= 1

	# def press_button(self, dest_floor):
	# 	if dest_floor > 0 and dest_floor < max_height:
	# 		self.dest_floors.append(dest_floor)

	def empty(self):
		self.curr_people = []
		self.curr_floor = 0


	# function loads people into the elevator
	def load_people(self, people_in_floor):
		res = 0
		if len(people_in_floor) > (self.max_people - len(self.curr_people)):
			res = people_in_floor[self.max_people - len(self.curr_people):]
			for p in people_in_floor[:self.max_people - len(self.curr_people)]:
				self.curr_people.append(p)
		else:
			for p in people_in_floor:
				self.curr_people.append(p)
			# self.curr_people += people_in_floor
			res = []
		
		return res

		# dest_floor = np.random.randint(floor+1,self.max_height) if button.up == "^" else np.random.randint(0,floor-1)		
		# if not dest_floor in self.dest_floors:
		# 	self.dest_floors.append(dest_floor)
		# sorted(self.dest_floors)

	# function unloads people back into the building 
	def unload_people(self, people_in_floor, max_capacity):#max_capacity는 해당 층의 max capacity를 뜻함. ㅅㅂ 이해하기 쉽지않네
		num_in_floor = len(people_in_floor)
		#num_in_lift = len(self.curr_people)
		res = self.curr_people

		######
		self.off_people=0

		#승객 중 이 층에 내릴 사람 idx 저장 할 리스트,
		this_floor_people = []
		for i in range(len(self.curr_people)):
			if self.curr_people[i].dest == self.curr_floor:
				this_floor_people.append(i)

		#이 층에서 내릴 사람 있으면
		if len(this_floor_people) !=0:
			self.off_people = len(this_floor_people)
			this_floor_people.reverse()
			for i in this_floor_people:
				res.pop(i)

#origin
#		if self.curr_floor == 0 or num_in_floor + num_in_lift < max_capacity:
#			self.curr_people = []
#		else:
#			self.curr_people = res[0:num_in_floor + num_in_lift - max_capacity]
#			res = res[num_in_floor + num_in_lift - max_capacity:]

#		return res
		return self.off_people














