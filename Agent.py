###agent

import os
import keras.layers as layers
import keras.optimizers as optimizers
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np

env_iteration_number = 5
learning_rate = 0.0001
lmbda =0.95
eps_clip = 0.1
gamma = 0.98
class Value:
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.__make_network()
        self.__make_loss_function()
        
    def __make_network(self):
        input_layer = layers.Input(shape=(self.input_shape,))
        x = layers.Dense(256,activation = 'relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256,activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1)(x)
        self.model = Model(inputs = input_layer, outputs = x)
    def get_value(self,state):
        return self.model.predict(state)
    
    def __make_loss_function(self):
        
        value_output = self.model.output
        reward_placeholder = K.placeholder(shape=(None,1),name = 'reward')
        HUBER_DELTA = 0.5
        y = tf.identity(reward_placeholder)
        loss = K.abs(reward_placeholder - value_output)
        loss = K.switch(loss < HUBER_DELTA, 0.5 * loss ** 2 , HUBER_DELTA * (loss - 0.5 * HUBER_DELTA))
        loss = K.sum(loss)
        #loss = K.mean(K.square(reward_placeholder - value_output))
        
        optimizer = optimizers.Adam(learning_rate)
        update = optimizer.get_updates(loss =loss, params = self.model.trainable_weights)
        
        self.update_function = K.function(inputs = [self.model.input,\
                                                   reward_placeholder],\
                                         outputs = [] , updates = update)

        
        
class Agent():
    def __init__(self, building_height, elevator_nums, actions):
        self.elevator_nums = elevator_nums
        self.input_shape = building_height + elevator_nums * 2
        self.output_shape = actions
        
        self.actor = Actor(self.input_shape,self.output_shape, self.elevator_nums)
        self.value = Value(self.input_shape)
        self.memory = []
    def get_action(self, state):
        return self.actor.get_action(state)
    def put_data(self,data):
        self.memory.append(data)
        
    def memory_to_trainable(self):
        state_list, action_list, reward_list, next_state_list, prob_list, done_list = [],\
        [], [], [], [], []
        
        for data in self.memory:
            state, action, reward, next_state, prob, done = data
            
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            next_state_list.append(next_state)
            prob_list.append([prob])
            done = 0 if done else 1
            done_list.append([done])
        return np.array(state_list), np.array(action_list), np.array(reward_list),\
                np.array(next_state_list), np.array(prob_list), np.array(done_list)     
        
    def train(self):
        global x_1
        global x_2
        global x_3
        global y
        state,action,reward,next_state,prob,done_mask = self.memory_to_trainable()
        state = state.reshape(-1,self.input_shape)
        action = action.reshape(-1,self.elevator_nums)
        next_state = next_state.reshape(-1,self.input_shape)
        done_mask = done_mask.reshape(-1,1)
        prob = prob.reshape(-1,self.elevator_nums, self.output_shape)
        for i in range(env_iteration_number):
            #print('train iterate : ',i)
            #print('state shape ' , state.shape)
            #print('gamma', gamma)
            #print(self.value.get_value(next_state))
            #print(done_mask.shape)
            td_error = reward + gamma * self.value.get_value(next_state) * done_mask
            #print('td_error',td_error)
            delta = np.array(td_error - self.value.get_value(state))
            
            #print('delta',delta)
            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                #print('advantage',advantage)
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage = np.array(advantage_list).reshape(-1,1)
            
            #print('advantage',advantage)
            (self.value.update_function([state,advantage])) ###########test 0 train 1

            x_1,x_2,x_3,y = (self.actor.update_function([state,action,prob,advantage]))###########test 0 train 1
            #print('surr_1 : ',x_1) #surr_1,surr_2,ratio,loss
            #print('surr_2 : ',x_2) #now_action_select, before_action_select, ratio,loss
            #print('ratio : ',x_3) #now_action_prob_1[0],now_action_prob_1[1], before_action_prob ,loss
            #print('loss : ',y) #,, \
            #raise Exception() #for test
        self.memory = []

    def save(self, num):
        self.actor.model.save_weights("./model_weights/actor_"+str(num)+".h5")
        self.value.model.save_weights("./model_weights/value_"+str(num)+".h5")

    def reload(self,num):
        self.actor.model.load_weights("./model_weights/actor_"+str(num)+".h5")
        self.value.model.load_weights("./model_weights/value_"+str(num)+".h5")
class Actor:
    def __init__(self,input_shape, output_shape,elevator_num):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.elevator_num = elevator_num
        self.__make_network()
        self.__make_loss_function()
        
    def __make_network(self):
        input_layer = layers.Input(shape=(self.input_shape,))    
        x = layers.Dense(256, activation = 'relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        xs = [layers.Dense(self.output_shape, activation = 'softmax')(x)\
              for _ in range(self.elevator_num)]
        self.model = Model(inputs = input_layer, outputs = xs)
    
    def get_action(self,state):
        return self.model.predict(state)
    
    def __make_loss_function(self):
        before_action_prob = K.placeholder(shape = (None, self.elevator_num,self.output_shape),\
                                          name = 'before_action_prob')
        before_action = K.placeholder(shape = (None, self.elevator_num),\
                                          name = 'before_action',dtype = 'int64') ########

        advantage = K.placeholder(shape = (None,1), name ='advantage')
        
        y = tf.identity(before_action_prob) #for test
        now_action_prob_1 = self.model.output
        
        
        #now_action_prob_test_1 = now_action_prob_1[0]
        #now_action_prob_test_2 = now_action_prob_1[1]
        #now_action_prob = tf.stack([now_action_prob_test_1,now_action_prob_test_2],axis=1)
        now_action_prob = K.reshape(now_action_prob_1,(-1,self.elevator_num,self.output_shape))
        
        now_action_select = K.sum(now_action_prob * tf.squeeze(tf.one_hot(before_action,\
                                                                          depth=self.output_shape)) ,axis=-1) 
        before_action_select = K.sum(before_action_prob * tf.squeeze(tf.one_hot(before_action,\
                                                                                depth=self.output_shape)) ,axis=-1)

        #ratio = - K.mean(now_action_select / before_action_select)
        #now_action_select = K.reshape(now_action_select,(-1,1))
        #before_action_select = K.reshape(before_action_select,(-1,1))
        ratio =  (K.exp(K.log(now_action_select)- K.log(before_action_select)))
        surr_1 = advantage * ratio
        surr_2 = advantage * K.clip(ratio, 1-eps_clip, 1+eps_clip) 
        loss = -K.mean(K.minimum(surr_1,surr_2))
        optimizer = optimizers.Adam(lr = learning_rate)
        updates = optimizer.get_updates(loss = loss, params = self.model.trainable_weights)
        ##현재 확인점
        ### 1. surr_1,surr_2,ratio,loss ## surr_1, surr_2 존나다른데?
        ### 2. now_action_select, before_action_select, ratio,loss ### nowaction beforeaction 존나다름
        ### 3. now_action_prob_1[0],now_action_prob_1[1], before_action_prob,loss 
        #### : output이 update가 된상태로 처음부터나오게되는데 왜그런지 모르겠음.
        ####K.learning_phase() 제거하니까 됨
        ### 4. surr_1,surr_2,ratio,loss
        self.update_function = K.function(inputs = [self.model.input,before_action,\
                                       before_action_prob,advantage],\
                            outputs = [surr_1, surr_2,ratio,  loss], \
                            updates = updates)       #output is for test
####agenttate shape 