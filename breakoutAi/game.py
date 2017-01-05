import pygame, sys
import numpy as np
#from breakoutenv import BreakoutEnv
#from breakouttask import BreakoutTask

#from pybrain.rl.learners.valuebased import ActionValueTable
#from pybrain.rl.agents import LearningAgent
#from pybrain.rl.learners import Q
#from pybrain.rl.experiments import Experiment
#from pybrain.rl.explorers import EpsilonGreedyExplorer

black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]

block_width = 60
block_height = 12

fname = 'trainedQ_breakout'

resolution = 20
alpha = 0.7
l = 1

Q = np.zeros((640/resolution,480/resolution,640/resolution,3))

STATES = {
    'Alive':0,
    'Dead':-10,
    'Scores':1,
    'Hit':1
}

ACTIONS = {
    0:"Stay here",
    1:"Go left",
    2:"Go right"
}

#the game's variables
ball_radius = 10


paddle_width = 60
paddle_height = 20



class Brick():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, block_width, block_height)



class Breakout(object):


    def __init__(self):

        self.iteration = 0
        self.ball_x = 20
        self.ball_y = 450-ball_radius
        self.ball_speed_x = 3
        self.ball_speed_y = 5

        self.paddle_x = 20
        self.paddle_y = 450
        self.paddle_speed =1
        self.paddle_vec = 0
        self.com_vec = 0

        self.bricks = []
        self.command = 0

        pygame.init()
        self.score = 0

        for i in range(1,9):
            for j in range(1,5):
                temp = Brick(70*i-35,50+20*j)
                self.bricks.append(temp)

        #allows for holding of key
        pygame.key.set_repeat(1,0)

        self.screen = pygame.display.set_mode([640,480])
        self.myfont = pygame.font.SysFont("Arial",  30)


    def getPositions(self):
        """Return the position of the ball and paddle as array of four value"""
        positions = int(self.ball_y/resolution*100+self.ball_x/resolution*10+self.paddle_x/resolution)
        return positions

    def getReward(self):
        """Return the game state (as a reward)"""
        return self.current_reward

    def comInput(self,command):
        """perform action requested by the machine"""
        self.command = int(command)


    def update(self):

        self.paddle_x += self.paddle_vec + self.com_vec
        if self.paddle_x < 0:
            self.paddle_x = 0
            self.paddle_vec = 0
        if self.paddle_x > self.screen.get_width() - paddle_width:
            self.paddle_x = self.screen.get_width() - paddle_width
            self.paddle_vec = 0

        self.current_reward = STATES['Alive']
        ##MOVE THE BALL
        self.ball_y += self.ball_speed_y
        self.ball_x += self.ball_speed_x

        self.hitDetect()

    def hitDetect(self):
        ##COLLISION DETECTION
        ball_rect = pygame.Rect(self.ball_x-ball_radius, self.ball_y-ball_radius, ball_radius*2,ball_radius*2) #circles are measured from the center, so have to subtract 1 radius from the x and y
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, paddle_width, paddle_height)

        #check if the ball is off the bottom of the self.screen
        if self.ball_y > self.screen.get_height() - ball_radius:
            self.ball_speed_y = -self.ball_speed_y
            self.current_reward = STATES['Dead']

            self.iteration+=1
            self.ball_y = 450-ball_radius
            s = 'Iteration: '+repr(self.iteration) + ', max score: ' + repr(self.score)
            print(s)
            self.score = 0
            if len(self.bricks) == 0:
                self.initBricks()

        #for screen border
        if self.ball_y < ball_radius:
            self.ball_speed_y = -self.ball_speed_y
        if self.ball_x < ball_radius:
            self.ball_speed_x = -self.ball_speed_x
        if self.ball_x > self.screen.get_width() - ball_radius:
            self.ball_speed_x = -self.ball_speed_x

        #for paddle
        if ball_rect.colliderect(paddle_rect):
            self.ball_speed_y = -self.ball_speed_y
            self.current_reward = STATES['Hit']
            if len(self.bricks) == 0:
                self.initBricks()
        #for bricks
        for brick in self.bricks:
            if brick.rect.colliderect(ball_rect):
                self.score = self.score + 1
                self.bricks.remove(brick)
                self.ball_speed_y = - self.ball_speed_y
                #self.current_reward = STATES['Scores']


    def input(self):
        isPressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    isPressed = True
                    self.paddle_vec -= self.paddle_speed

                elif event.key == pygame.K_RIGHT:
                    isPressed = True
                    self.paddle_vec += self.paddle_speed

        if not isPressed:
            if self.paddle_vec >0:
                self.paddle_vec -= self.paddle_speed
            elif self.paddle_vec < 0:
                self.paddle_vec += self.paddle_speed

        return True

    def decision(self):

        self.prev = [self.ball_x/resolution,self.ball_y/resolution,self.paddle_x/resolution]
        #Observe what state is in and perform the action that maximizes expected reward.
        actions = Q[self.ball_x/resolution,self.ball_y/resolution,self.paddle_x/resolution,:]
        self.command = np.argmax(actions)


        if self.command == 1:
            #self.com_vec -= self.paddle_speed
            self.paddle_x -= self.paddle_speed
        elif self.command == 2:
            #self.com_vec += self.paddle_speed
            self.paddle_x += self.paddle_speed
        else:
            if self.com_vec >0:
                #self.com_vec -= self.paddle_speed
                self.paddle_x -= self.paddle_speed
            elif self.com_vec < 0:
                #self.com_vec += self.paddle_speed
                self.paddle_x += self.paddle_speed
    def observe(self):
        prev_Q = Q[self.prev[0],self.prev[1],self.prev[2],self.command]

        Q[self.prev[0],self.prev[1],self.prev[2],self.command] = (
            prev_Q + alpha * (self.current_reward + l *
                              max(Q[self.ball_x/resolution,self.ball_y/resolution,self.paddle_x/resolution,:])
                              - prev_Q))




    def draw(self):
        #DRAW EVERYTHING
        #pygame.time.delay(1)
        self.screen.fill(white)

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (5, 10))
        for brick in self.bricks:
            pygame.draw.rect(self.screen,grey,brick.rect,0)
        pygame.draw.circle(self.screen, grey, [int(self.ball_x), int(self.ball_y)], ball_radius, 0)
        pygame.draw.rect(self.screen, grey, [self.paddle_x, self.paddle_y, paddle_width, paddle_height], 0)

        #update the entire display
        pygame.display.update()


    def quit(self):
        #save the Trained Q before quit python
        np.save(fname,Q)
        pygame.quit()

    def initBricks(self):
        for i in range(1,9):
            for j in range(1,5):
                temp = Brick(70*i-35,50+20*j)
                self.bricks.append(temp)

game = Breakout()
#env = BreakoutEnv(game)
#task = BreakoutTask(env)

#av_table = ActionValueTable(688, 3)
#av_table.initialize(0.)

#learner = Q(alpha=0.5,gamma=0.99)
#learner._setExplorer(EpsilonGreedyExplorer(epsilon = 0.3))
#agent = LearningAgent(av_table, learner)

#experiment = Experiment(task, agent)

try:
    Q = np.load(fname + '.npy')
    print("Q loaded successfully.")
except:
    print("Unexpected error:", sys.exc_info()[0])


#game loop
while game.input():
    game.decision()
    game.update()
    game.observe()
    #experiment.doInteractions(1)
    #agent.learn()
    game.draw()

game.quit()
