import pygame
from math import sqrt
import numpy as np

class Body:
    def __init__(self, x, y, mass=1, velocity=np.array([0,0],dtype='float64'), color=(0,0,0), radius=5):
        self.position = np.array([x,y],dtype='float64')
        self.mass = mass
        self.velocity = velocity
        self.color = color
        self.acc = np.zeros(2)
        self.radius = radius
        
    def distance(self,P):
        displacement = self.position - P.position
        distance = sqrt((displacement[0])**2+(displacement[1])**2)
        return distance, displacement
    
    def isEqualTo(self,P,tol=1e-16):
        if abs(self.position[0] - P.position[0]) < tol and abs(self.position[1] - P.position[1]) < tol:
            return True
        else:
            return False
        
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.position[0], self.position[1]) + ' mass={:.2f}'.format(self.mass)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.position[0] , surface.get_height() - self.position[1]), self.radius)

class Square:
    def __init__(self,cx,cy,side):
        self.cx, self.cy = cx, cy
        self.side = side
        self.left, self.right = cx - side/2, cx + side/2
        self.bottom, self.top = cy - side/2, cy + side/2
                
    def contains(self,P):
        if P.position[0] >= self.left and P.position[0] <= self.right and P.position[1]<=self.top and P.position[1]>=self.bottom:
            return True
        else:
            return False      

    def draw(self, surface, depth):
        pygame.draw.rect(surface, (255,255,255), pygame.Rect(self.left , surface.get_height() - self.top , self.side, self.side), 2)