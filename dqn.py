from Env import Env 
import random
import pylab
import numpy as np
import tensorflow as tf
from collections import deque
import copy
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
        
        W1 = tf.Variable(tf.random_uniform([self.state_size, 32 ], 0, 0.01)) # Variable은 학습시킬 변수의 자리
        W2 = tf.Variable(tf.random_uniform([32 , 16 ], 0, 0.01))
        W3 = tf.Variable(tf.random_uniform([16 , 16 ], 0, 0.01))
        W4 = tf.Variable(tf.random_uniform([16 , self.action_size], 0, 0.01))
        
        # 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
        # b1 은 히든 레이어의 뉴런 갯수로, b2 는 최종 결과값 즉, 분류 갯수인 3으로 설정합니다.
        b1 = tf.Variable(tf.zeros([32 ]))
        b2 = tf.Variable(tf.zeros([16 ]))
        b3 = tf.Variable(tf.zeros([16 ]))
        
        # 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
        L1 = tf.add(tf.matmul(X, W1), b1)
        L1 = tf.nn.relu(L1)
        
        # 신경망의 히든 레이어에 가중치 W2과 편향 b2을 적용합니다
        L2 = tf.add(tf.matmul(L1, W2), b2)
        L2 = tf.nn.relu(L2)
        
        L3 = tf.add(tf.matmul(L2, W3), b3)
        L3 = tf.nn.relu(L3)
        Qpredict = tf.matmul(L3, W4) 
        return Qpredict

    # get action from model using epsilon-greedy policy
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
    X = tf.placeholder(shape=[1, state_size], dtype=tf.float32) # placeholder는 데이터 input이 들어가는 자리
    Qpredict = agent.build_model(X)
    Y = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(Y-Qpredict)) # 오차 제곱의 합
    train = tf.train.AdamOptimizer(learning_rate=agent.learning_rate).minimize(loss) # 경사하강법을 사용하여 오차를 최소화하는 방향으로 학습을 시키겠다
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
                Qs = sess.run(Qpredict, feed_dict={X:state}) # feed_dict은 placeholder에 실제 값을 넣기 위해 설정하는 옵션
                action = agent.get_action(state,Qs,len(env.snake)-1)
                next_state, reward, done,dead = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.avg_q_max += np.amax(Qs)


                # save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(state, action, reward, next_state, dead)
                eIdx = len(env.snake)-1
                if agent.epsilon[eIdx] > agent.epsilon_end:
                    agent.epsilon[eIdx] -= agent.epsilon_decay_step
                # every some time interval, train model

                score += reward

                # if agent is dead, then reset the state
                if dead:
                    dead = False
                else:
                    state = next_state
                
                    

                # if done, plot the score over episodes
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
                    
            if e % 10 == 1:
                if len(agent.memory) < agent.train_start:
                    continue
                for j in range(MINIBATCH):
                    # 메모리에서 사용할 리플레이를 랜덤하게 가져옴
                    for sample in random.sample(agent.memory, 50):
                        state_r, action_r, reward_r, new_state_r, dead_r = sample
                        Qs = sess.run(Qpredict, feed_dict={X: state_r})
                        # DQN 알고리즘으로 학습
                        if dead_r:
                            Qs[0, action_r] = reward_r
                        else:
                            new_Qs = sess.run(Qpredict, feed_dict={X: new_state_r})
                            Qs[0, action_r] = reward_r + agent.discount_factor * np.max(new_Qs)
                        sess.run(train, feed_dict={X: state_r, Y: Qs})
            if e% 1000 == 1:
                f = open("epi_score.txt","w")
                for i,j in zip(episodes,scores):
                    f.write("%d, %d\n"%(i,j))
                f.close()


