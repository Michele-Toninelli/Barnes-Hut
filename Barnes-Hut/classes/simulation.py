import pygame
import time
import sys
import numpy as np
from classes.system_naive import System_Naive
from classes.quadtree_numba import System_Quadtree_Numba
from classes.quadtree import System_Quadtree

import os
import time
from numba import njit
FPS = 2000
DELTA_T = 0.0005

def empty_folder():
    for f in os.listdir('frames'):
        os.remove('frames/'+f)


class Simulation:
    def __init__(self,position, mass, velocity, acceleration, color, cx, cy, side, max_points, theta, mode, saveFrame):
        self.run = False
        self.clock = pygame.time.Clock()

        self.position = position
        self.mass = mass
        self.velocity = velocity
        self.acceleration = acceleration
        self.color = color

        self.cx = cx
        self.cy = cy
        self.side = side
        self.max_points = max_points
        self.theta = theta
        self.saveFrames = saveFrame

        self.mode = mode

        self.initialise_environment()
        self.show_environment()          

    def initialise_environment(self):
        space_plane_size = (self.side,self.side)  # width, height of canvas
        self.run = True

        # setting up pygame
        pygame.init()
        pygame.display.set_caption("N-body simulation")
        self.surface = pygame.display.set_mode(space_plane_size)
        self.start = time.perf_counter()

        if self.saveFrames:
            empty_folder()

        if self.mode == 0:
            self.system = System_Quadtree_Numba(self.position, self.mass, self.velocity, self.acceleration, self.color, self.cx,self.cy,self.side,self.max_points,self.theta,self.surface)
        elif self.mode == 1:
            self.system = System_Quadtree(self.position, self.mass, self.velocity, self.color, self.cx,self.cy,self.side,self.max_points,self.theta,self.surface)
        elif self.mode == 2:
            self.system = System_Naive(self.position, self.mass, self.velocity, self.color, self.surface)


    def show_environment(self):
        iter=0
        filename = 0
        self.iters_exe = []
        while self.run:
            begin_iter = time.perf_counter()
            # background
            self.surface.fill((0,0,0))
            #button
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if  event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.run = False

            self.system.computeNewFrame()           

            pygame.display.update()
            iter+=1
            if self.saveFrames and iter%100==0:
                filename+=1
                pygame.image.save(self.surface,'frames/'+str(filename)+'.png')
            
            end_iter = time.perf_counter()
            self.iters_exe.append(end_iter-begin_iter)
            self.clock.tick(FPS)
        self.quit()

    def quit(self):
        self.end = time.perf_counter()
        print('exe time: ',self.end-self.start)
        self.iters_exe = self.iters_exe[1:] #remove first iteration
        print('n iters ',len(self.iters_exe),' mean exe time ',np.mean(self.iters_exe),' min ',np.min(self.iters_exe),' max ',np.max(self.iters_exe))
        pygame.quit()
        sys.exit()