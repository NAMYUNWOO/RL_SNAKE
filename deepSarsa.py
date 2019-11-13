from Env import Env 
import pylab
import random
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop,Adagrad
from tensorflow.keras.layers import Dense, Flatten,Conv2D
#from keras import backend as K
from collections import deque

EPISODES = 5000000

class SnakeAgent:
    def __init__(self,state_size):
        tf.keras.backend.clear_session()
        self.state_size = state_size
        self.action_size = 4
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.exploration_steps = 50000.
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 500
        self.update_target_rate = 10000
        self.discount_factor = 0.90
        self.learning_rate = 0.001
        self.load_model = False        
        if self.load_model:
            self.model.load_weights("./save_model/deepsarsa.h5")
        #self.sess = tf.compat.v1.InteractiveSession() 
        #K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        #self.summary_placeholders, self.update_ops, self.summary_op =  self.setup_summary()
        #self.summary_writer = tf.compat.v1.summary.FileWriter( 'summary/deepsarsa', self.sess.graph)
        #self.sess.run(tf.compat.v1.global_variables_initializerz())            

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size[0]//2, activation='relu', input_shape=self.state_size))
        model.add(Dense(self.state_size[0]//4, activation='relu'))
        model.add(Dense(self.state_size[0]//8, activation='relu'))
        model.add(Dense(self.action_size,activation="linear"))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=0.00005))
        return model
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.float32(state / 255.0)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])    


    def train_model(self,state,action,reward,next_state,next_action,done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        state = np.float32(state / 255.0)
        next_state  = np.float32(next_state / 255.0)
        target = self.model.predict(state)[0]

        if done:
            target[action] = reward
        else:
            target[action] = reward+self.discount_factor*self.model.predict(next_state)[0][next_action]
        
        target = np.reshape(target,[1,self.action_size])
        self.model.fit(state,target,epochs=1,verbose=0)
