from Env import Env
import numpy as np
import os,copy
import time
from deepSarsa import SnakeAgent as SA
clearCmd = "clear"
EPISODES = 500000
def deepsarsa():
    global clearCmd
    os.system(clearCmd)
    render=True
    env =Env(width_height=7,frame_size=2,isSummary=False)
    #state_size = env.state_size
    state_ele_size = 7*7
    state_size = (7*7*2,) # 하드코딩함
    frame_size = env.frame_size
    
    snakeAgent = SA(state_size)
    filehist = open("./history_deepsarsa.txt",'w')
    for e in range(EPISODES):
        state = env.reset()
        state_,_,_,_ = env.step(env.myAction)
        state = state.reshape(state_ele_size)
        state = np.concatenate([state,state])
        step,score = 0,0
        done = False
        while not done:
            time.sleep(0.2)
            state_ = copy.deepcopy(state)
            action  = snakeAgent.get_action(np.array([state_]))
            next_state,reward,done,dead = env.step(action)
            next_state = next_state.reshape(state_ele_size)
            next_state = np.concatenate([next_state,state_[:-len(state_)//frame_size ]])
            next_action = snakeAgent.get_action(np.array([next_state]))
            snakeAgent.train_model(np.array([state_]),
                                    action,
                                    reward,
                                    np.array([copy.deepcopy(next_state)]),
                                    next_action,
                                    done
                                  )
            if dead:
                dead = False
            else:
                state = next_state
            #state = next_state
            score += reward
            step += 1
            if render:
                env.render()
            print(score)
        filehist.write("epi:{}, score:{}, step:{} \n".format(e,score,step))

def userGame():
    global clearCmd
    os.system(clearCmd)
    print("please, sudo required")
    print("control keys: W A S D")
    input("push anykey game start")
    import threading
    os.system(clearCmd)
    env = Env(11)
    keygetting = threading.Thread(target=env.keypressInput)
    keygetting.start()
    env.reset()
    score = 0
    while True:
        time.sleep(0.2)
        next_state,reward,done,dead = env.step(env.myAction)
        score += reward
        env.render()
        print(score)
        if done:
            break
    
    print("game end")

if __name__ == "__main__" :

    if os.name != "posix":
        clearCmd = "cls"
    opt = ""
    while True:
        opt = input("choose option 1 or 2\n1) playing game\n2) learing RL\n").strip()
        if opt not in ["1",'2']:
            print("choose 1 or 2")
        else:
            break
    
    if opt == "1":
        while True:
            userGame()
            input("press anykey to restart")
    else:
        deepsarsa()

