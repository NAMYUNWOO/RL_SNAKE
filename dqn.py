from Env import Env 
import random
import pylab
import numpy as np
import tensorflow as tf
from collections import deque
import copy
pylab
EPISODES = 500000
MINIBATCH = 50


class DQNAgent:
    def __init__(self, state_size,action_size,render,snakeMax):
        self.render = render
        self.load_model = False
        # environment settings
        self.state_size =  state_size
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = np.ones(snakeMax)
        self.epsilon_start, self.epsilon_end = 1.0, 0.001
        self.exploration_steps = 100000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.learning_rate = 0.0001
        self.batch_size = 50
        self.train_start = 1000
        self.discount_factor = 0.90
        self.memory = deque(maxlen=2000)
        self.sess = tf.InteractiveSession()
        self.avg_q_max = 0  
        

    def build_model(self,X):
        
        W1 = tf.Variable(tf.random_uniform([self.state_size, 32 ], 0, 0.01))
        W2 = tf.Variable(tf.random_uniform([32 , 16 ], 0, 0.01))
        W3 = tf.Variable(tf.random_uniform([16 , 16 ], 0, 0.01))
        W4 = tf.Variable(tf.random_uniform([16 , self.action_size], 0, 0.01))
        
    
        b1 = tf.Variable(tf.zeros([32 ]))
        b2 = tf.Variable(tf.zeros([16 ]))
        b3 = tf.Variable(tf.zeros([16 ]))
        
        L1 = tf.add(tf.matmul(X, W1), b1)
        L1 = tf.nn.relu(L1)
        
        L2 = tf.add(tf.matmul(L1, W2), b2)
        L2 = tf.nn.relu(L2)
        
        L3 = tf.add(tf.matmul(L2, W3), b3)
        L3 = tf.nn.relu(L3)
        Qpredict = tf.matmul(L3, W4) 
        return Qpredict

    def get_action(self,state,Qs,eIdx):
        if np.random.rand() <= self.epsilon[eIdx]:
            return random.randrange(self.action_size)
        else:
            return np.argmax(Qs)

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, dead):
        self.memory.append((state, action, reward, next_state, dead))

if __name__ == "__main__":
    width_height = 7
    env = Env(width_height=width_height,frame_size=1,isSummary=True)
    state_size = env.state_size
    action_size = env.action_size
       
    agent = DQNAgent(state_size=state_size,action_size=action_size,render=False,snakeMax=width_height**2)
    X = tf.placeholder(shape=[None, state_size], dtype=tf.float32) 
    Qpredict = agent.build_model(X)
    Y = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(Y-Qpredict)) 
    train = tf.train.AdamOptimizer(learning_rate=agent.learning_rate).minimize(loss) 
    scores, episodes, global_step = [], [], 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPISODES):
            done = False
            dead = False
            
            step, score = 0, 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            
            while not done:
                if agent.render:
                    env.render()
                global_step += 1
                step += 1
                Qs = sess.run(Qpredict, feed_dict={X:state}) 
                action = agent.get_action(state,Qs,len(env.snake)-1)
                next_state, reward, done,dead = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.avg_q_max += np.amax(Qs)


                agent.replay_memory(state, action, reward, next_state, dead)
                eIdx = len(env.snake)-1
                if agent.epsilon[eIdx] > agent.epsilon_end:
                    agent.epsilon[eIdx] -= agent.epsilon_decay_step

                score += reward

                if dead:
                    dead = False
                else:
                    state = next_state
                
                    

                if done:
                    if agent.render:
                        continue 
                    if global_step > agent.train_start:
                        stats = [score, agent.avg_q_max / float(step), step,
                                agent.avg_loss / float(step)]
                    scores.append(score)
                    episodes.append(e)
                    print("episode:", e, "  score:", score, "  memory length:",
                        len(agent.memory), "  epsilon:",round(agent.epsilon[len(env.snake)-1],4),
                        "  global_step:", global_step, "  average_q:",
                        agent.avg_q_max / float(step), "snake Length:",len(env.snake))

                    agent.avg_q_max, agent.avg_loss = 0, 0
                    
            
                if len(agent.memory) < agent.train_start:
                    continue
                
                minibatch =  random.sample(agent.memory, agent.batch_size)
                states = np.zeros([agent.batch_size,state_size])
                next_states = np.zeros([agent.batch_size,state_size])
                actions, rewards, deads = [], [], []
                for i in range(agent.batch_size):
                    states[i] = minibatch[i][0]
                    next_states[i] = minibatch[i][3]
                    actions.append(minibatch[i][1])
                    rewards.append(minibatch[i][2])
                    deads.append(minibatch[i][4])
                target = sess.run(Qpredict, feed_dict={X: states}) 
                target_value = sess.run(Qpredict, feed_dict={X: next_states})
                for i in range(agent.batch_size):
                    if deads[i]:
                        target[i][actions[i]] = rewards[i]
                    else:
                        target[i][actions[i]] = rewards[i] + agent.discount_factor* np.amax(target_value[i])
                sess.run(train, feed_dict={X: states, Y: target}) 
                """
                
                for sample in random.sample(agent.memory, 50):
                    state_r, action_r, reward_r, new_state_r, dead_r = sample
                    Qs = sess.run(Qpredict, feed_dict={X: state_r})
                    if dead_r:
                        Qs[0, action_r] = reward_r
                    else:
                        new_Qs = sess.run(Qpredict, feed_dict={X: new_state_r})
                        Qs[0, action_r] = reward_r + agent.discount_factor * np.max(new_Qs)
                    sess.run(train, feed_dict={X: state_r, Y: Qs})
                """
                    
            if e% 10000 == 1:
                pylab.plot(episodes,scores,'b')
                pylab.savefig("./save_graph/breakout_dqn_.png")
                saver = tf.train.Saver()
                saver.save(sess, './save_model/dqn.ckpt')


