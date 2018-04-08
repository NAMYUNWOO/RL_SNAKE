from collections import deque
import random
import numpy as np
import os
import threading
import curses
from pynput import keyboard
import time
ACTIONS = [[-1,0],[0,1],[1,0],[0,-1]]
class Env:

    def __init__(self,width_height=11,frame_size=4,isSummary=False):
        """
        observation_space = serialized game Field * 4 frames
        
        """
        self.isSummary = isSummary
        self.groundVal = 0
        self.appleVal = -255
        self.tailVal = 150
        self.headVal = 255
        self.width_height = width_height
        self.frame_size = frame_size 
        if self.isSummary:
            self.state_size = 6
        else:
            if frame_size == 1:
                self.state_size = width_height*width_height
            else:
                self.state_size = (width_height,width_height,self.frame_size)
        self.action_size = len(ACTIONS)
        self.snake = set()
        self.fieldXYset = set([(i,j) for i in range(self.width_height) for j in range(self.width_height)])
        self.snakeGrad = list(range(255,10,-(255-10)//(width_height*width_height)))
        self.inputKey = ""
        self.myAction = random.randint(0,3)
        self.eatSelf = set()
        self.appleLoc = (0,0)
        
    
    def getBarrierDists(self,head):
        hx,hy = head[0],head[1]
        dist = [hy+1,self.width_height+1-hx,self.width_height+1-hy,hx+1] # N, E, S, W

        

        for dist_i,i in enumerate(range(hy-1,-1,-1)):
            if self.state[hx,i] == -1:
                dist[0] = dist_i + 1 
                break

        for dist_i,i in enumerate(range(hx+1,self.width_height)):
            if self.state[i,hy] == -1:
                dist[1] = dist_i + 1
                break
        for dist_i,i in enumerate(range(hy+1,self.width_height)):
            if self.state[hx,i] == -1:
                dist[2] = dist_i + 1
                break
        for dist_i,i in enumerate(range(hx-1,-1,-1)):
            if self.state[i,hy] == -1:
                dist[3] = dist_i + 1
                break
        return dist

    def nearCheck(self,head):
        hx,hy = head[0],head[1]
        nears = []
        for i,j in ACTIONS:
            checkBack = np.array(ACTIONS[self.myAction])+np.array([i,j])
            if int(checkBack[0]) == 0 and int(checkBack[1]) == 0:
                nears.append(0)
                continue
            ii = hx + i
            jj = hy + j
            if ii < 0 or jj < 0 or ii >= self.width_height or jj >= self.width_height:
                nears.append(0)
            else:
                if self.state[ii][jj] >= self.tailVal:
                    nears.append(0)
                else:
                    nears.append(1)
        return nears



    def getSummarizedState(self):
        head = self.snake[0]
        #barrierDists = self.getBarrierDists(head)
        nears = self.nearCheck(head)
        summarizedState = np.array(nears + [self.appleLoc[0]-head[0],self.appleLoc[1]-head[1]])
        #print(summaryState)
        #summarizedState.dtype = np.float
        return np.float32(np.array([summarizedState]))


    def reset(self):
        self.snake = deque([((self.width_height//2),(self.width_height//2))])
        self.state = np.full((self.width_height,self.width_height),self.groundVal)
        self.state[self.width_height//2,self.width_height//2] = self.headVal  #init snake loc
        self.putApple(True)
        if self.isSummary:
            returnState = self.getSummarizedState()
        else:
            returnState = self.state
        return returnState
    
    def getAppleLoc(self):
        emptySpace = self.fieldXYset - set(self.snake)
        if len(self.snake) == 1:
            for i,j in ACTIONS:
                emptySpace =emptySpace-set([self.snake[0][0]+i,self.snake[0][1]+j])

        emptySpace = list(emptySpace)
            
        randomN = np.random.randint(len(emptySpace))
        return emptySpace[randomN] 

    def render(self):
        os.system("clear")
        print("& "*(self.width_height+2))
        for row in self.state:
            print("& ",end="")
            for i in row:
                if i == self.groundVal: print("  ",end="")    # ground
                elif i == self.appleVal : print("o ",end="")  # apple 
                elif i < self.headVal : print("□ ",end="")  # tail
                else: print("■ ",end="")                    # head
            print("& ")
        print("& "*(self.width_height+2))
        
    
    def putApple(self,newApple):
        if not newApple:
            return
        else:
            self.appleLoc = self.getAppleLoc()
            self.state[self.appleLoc[0]][self.appleLoc[1]] = self.appleVal
            return

    def step(self,action):
        i,j = self.snake[0]
        v,h = ACTIONS[action]
        reward = 0
        done = False
        dead = False
        newApple = False
        backCheck = np.array(ACTIONS[self.myAction]) + np.array(ACTIONS[action])
        if i+v < 0 or i+v >= self.width_height or j+h < 0 or j+h >= self.width_height:
            # snake hits border 
            reward = -10.0
            done = True
            dead = True
        elif int(backCheck[0])== 0 and int(backCheck[1]) == 0:
            # back direction
            reward = -10.0
            done = True
            dead = True
        else:
            self.snake.appendleft((i+v,j+h))
            if self.state[i+v][j+h] == self.groundVal:
                # step to empty ground
                #self.eatSelf = set()
                reward = 0.0
                ii,jj = self.snake.pop()
                self.state[ii][jj] = 0
            elif self.state[i+v][j+h] == self.appleVal:
                # step to Apple location (eat apple)
                #self.eatSelf = set()
                #reward = 10.0*(len(self.snake)-1)
                reward = 10
                newApple = True
            else:
                # step to self location (eat self)
                
                reward = -10.0
                dead = True
                done = True
                ii,jj = self.snake.pop()
                self.state[ii][jj] = 0
            self.state[i+v][j+h] = self.headVal
            if len(self.snake) >= 2:
                self.state[i][j] = self.tailVal
        
        
        self.myAction = action
        if self.fieldXYset - set(self.snake) == set():
            reward = 100
            done = True
        self.putApple(newApple)
        if self.isSummary:
            returnState = self.getSummarizedState()
        else:
            returnState = self.state
        return returnState,reward,done,dead

    def on_press(self,key):
        try:
            myInput = key.char
            if myInput == "w":
                self.myAction = 0
            elif myInput == "d": 
                self.myAction = 1
            elif myInput == "s":
                self.myAction = 2
            elif myInput == "a":
                self.myAction = 3
        except AttributeError:
            pass

    def on_release(self,key):
        return

    def keypressInput(self):
        with keyboard.Listener( on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
