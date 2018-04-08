from Env import Env 
import pylab
import random
import copy
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam,RMSprop,Adagrad
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 5000000

class SnakeAgent:
    def __init__(self,state_size):
        self.state_size = state_size
        self.action_size = 4
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.model.load_weights("./save_model/breakout_dqn.h5")

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size,activation="linear"))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=0.00005))
        return model
    def get_action(self, history):
        history = np.float32(history / 255.0)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0])





class DQNAgent:
    def __init__(self, state_size,action_size,render):
        self.render = render
        self.load_model = False
        self.state_size =  state_size
        self.action_size = action_size
        self.epsilon = np.ones(state_size[0]*state_size[1])
        self.epsilon_start, self.epsilon_end = 1.0, 0.001
        self.exploration_steps = 50000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 64
        self.train_start = 500
        self.update_target_rate = 10000
        self.discount_factor = 0.90
        self.learning_rate = 0.001
        self.memory = deque(maxlen=200000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn6_.h5")
        self.update_target_model()
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter( 'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())



    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(288, activation='relu'))
        model.add(Dense(self.action_size,activation="linear"))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=0.00005))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history,eIdx):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon[eIdx]:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_replay(self,eIdx):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon[eIdx] > self.epsilon_end:
            self.epsilon[eIdx] -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target = self.model.predict(history)
        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if dead[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])
        self.model.fit(history, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def save_model(self, name):
        self.model.save_weights(name)

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


if __name__ == "__main__":
    env = Env(width_height=7,frame_size=4,isSummary=False)
    state_size = env.state_size
    frame_size = env.frame_size
    action_size = env.action_size
       
    agent = DQNAgent(state_size=state_size,action_size=action_size,render=False)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        
        step, score = 0, 0
        state = env.reset()
        state_,_,_,_ = env.step(env.myAction)
        hi,hj = env.snake[0]
        history = np.stack([state_]+[state for _ in range(frame_size-1)], axis=2)
        history = np.reshape([history], (1, state_size[0], state_size[1], state_size[2]))
        
        while not done:
            
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            history_ = copy.deepcopy(history)
            history_[0,hi,hj,0] *= 2
            action = agent.get_action(history_,len(env.snake)-1)
            
            

            next_state, reward, done,dead = env.step(action)
            hi,hj = env.snake[0]
            
            next_state = np.reshape([next_state], (1, state_size[0], state_size[1], 1))

            
            if reward > 0:
                next_history = np.stack(tuple(next_state for _ in range(frame_size)), axis=2)        
                next_history = np.reshape([next_history], (1, state_size[0], state_size[1], state_size[2]))
            else:
                next_history = np.append(next_state, history[:, :, :, :frame_size-1], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])
            next_history_ = copy.deepcopy(next_history)
            next_history_[0,hi,hj,0] *= 2
            agent.replay_memory(history_, action, reward, next_history_, dead)
            agent.train_replay(len(env.snake)-1)
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history
            
                

            if done:
                if agent.render:
                    continue 
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", round(agent.epsilon[len(env.snake)-1],4),
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step),"snake Length:",len(env.snake))

                agent.avg_q_max, agent.avg_loss = 0, 0
                
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn_.h5")
            pylab.plot(episodes,scores,'b')
            pylab.savefig("./save_graph/breakout_dqn_.png")
