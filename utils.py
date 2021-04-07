import torch
import numpy as np
class Rollouts(object):
    def __init__(self):
        self.rollouts = []
    def append(self,transition):
        self.rollouts.append(transition)
    def make_batch(self,device):
        state_1_list,state_2_list,state_3_list, action_list, reward_list, next_state_1_list,next_state_2_list,next_state_3_list, prob_list, done_list = [],[],[],[],[],[],[],[],[],[]
        for data in self.rollouts:
            state_1,state_2,state_3,action,reward,next_state_1,next_state_2,next_state_3,prob,done = data
            state_1_list.append(state_1)
            state_2_list.append(state_2)
            state_3_list.append(state_3)
            action_list.append(action)
            reward_list.append([reward])
            prob_list.append(prob)
            next_state_1_list.append(next_state_1)
            next_state_2_list.append(next_state_2)
            next_state_3_list.append(next_state_3)
            done_mask = 0 if done else 1
            done_list.append([done_mask])
        self.rollouts = []
        
        s1,s2,s3,a,r,next_s1,next_s2,next_s3,done_mask,prob = torch.tensor(state_1_list,dtype=torch.float).to(device),\
                                            torch.tensor(state_2_list,dtype=torch.float).to(device),\
                                            torch.tensor(state_3_list,dtype=torch.float).to(device),\
                                        torch.tensor(action_list).to(device),torch.tensor(reward_list).to(device),\
                                        torch.tensor(next_state_1_list,dtype=torch.float).to(device),\
                                        torch.tensor(next_state_2_list,dtype=torch.float).to(device),\
                                        torch.tensor(next_state_3_list,dtype=torch.float).to(device),\
                                        torch.tensor(done_list,dtype = torch.float).to(device),\
                                        torch.tensor(prob_list).to(device)
        return s1.squeeze(1),s2.squeeze(1),s3.squeeze(1),a,r,next_s1.squeeze(1),next_s2.squeeze(1),next_s3.squeeze(1),done_mask,prob
    
    def choose_mini_batch(self, mini_batch_size, states1,states2,states3, actions, rewards, next_states1,next_states2,next_states3, done_mask, old_log_prob, advantages, returns,old_value):
        full_batch_size = len(states1)
        full_indices = np.arange(full_batch_size)
        np.random.shuffle(full_indices)
        for i in range(full_batch_size // mini_batch_size):
            indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
            yield states1[indices],states2[indices],states3[indices], actions[indices], rewards[indices], next_states1[indices],next_states2[indices],next_states3[indices], done_mask[indices],\
                  old_log_prob[indices], advantages[indices], returns[indices],old_value[indices]
            
def is_finish(state):
    finish_check_1 = (state[0][0][0][0] == -0.1)
    finish_check_2 = (state[1][0][:,0] == -0.1).all()
    return (finish_check_1 and finish_check_2)

def state_preprocessing(args,device,floor_state,elv_state,elv_place_state):
    floor_state = torch.tensor(floor_state).transpose(1,0).unsqueeze(0).float()
    floor_state = torch.cat((floor_state,-1* torch.ones((1,2,args.building_height*args.max_people_in_floor- floor_state.shape[2]))),-1)/10.
    elv_state = [elv_state[idx]+([-1] * (args.max_people_in_elevator- len(elv_state[idx]))) for idx in range(len(elv_state))]
    elv_state = torch.tensor(elv_state).unsqueeze(0).float()/10.
    
    elv_place_state = torch.tensor(elv_place_state).unsqueeze(0).float()/10.
    return floor_state.to(device),elv_state.to(device),elv_place_state.to(device)

