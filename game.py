from Env import Env
import numpy as np
import os,copy
import time
from breakout_dqn import SnakeAgent as SA

def breakout_dqn_game():
    os.system("clear")
    env =Env(width_height=7,frame_size=4,isSummary=False)
    state_size = env.state_size
    frame_size = env.frame_size
    snakeAgent = SA(state_size)
    state = env.reset()
    state_,_,_,_ = env.step(env.myAction)
    hi,hj = env.snake[0]
    history = np.stack([state_]+[state for _ in range(frame_size-1)], axis=2)
    history = np.reshape([history], (1, state_size[0], state_size[1], state_size[2]))
    score = 0
    while True:
        time.sleep(0.2)
        history_ = copy.deepcopy(history)
        history_[0,hi,hj,0] *= 2
        action  = snakeAgent.get_action(history_)
        next_state,reward,done,dead = env.step(action)
        hi,hj = env.snake[0]
        next_state = np.reshape([next_state], (1, state_size[0], state_size[1], 1))
        next_history = np.append(next_state, history[:, :, :, :frame_size-1], axis=3)
        history = next_history
        score += reward
        env.render()
        print(score)
        if done:
            break

def userGame():
    os.system("clear")
    print("please, sudo required")
    print("control keys: W A S D")
    input("push anykey game start")
    import threading
    os.system("clear")
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
    opt = ""
    while True:
        opt = input("choose option 1 or 2\n1)playing game\n2)learing RL").strip()
        if opt not in ["1",'2']:
            print("choose 1 or 2")
        else:
            break
    
    if opt == "1":
        userGame()
    else:
        breakout_dqn_game()

