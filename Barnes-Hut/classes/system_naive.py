import numpy as np
from classes.quadtree_numba import _compute_acceleration
from classes.classes import Body
DELTA_T = 0.0005

class System_Naive:
    def __init__(self, position, mass, velocity, color, surface):
        self.bodies = self.initialize_bodies(position, mass, velocity, color)
        self.surface = surface

    def initialize_bodies(self, position, mass, velocity, color):
        bodies = []
        for i in range(len(position)):
            body = Body(position[i][0],position[i][1],mass[i],velocity[i],color[i],1)
            bodies.append(body)
        return bodies

    def acc_onBody(self,body):
        acc = 0
        for otherBody in self.bodies:
            if not otherBody.isEqualTo(body):
                r, d = body.distance(otherBody)
                acc += _compute_acceleration(r,d,body.radius,otherBody.mass)
        body.acc = acc

    def computeNewFrame(self):
        #The positions and velocities are updated using a leap-frog scheme
        for body in self.bodies:
            # (1/2) kick
            body.velocity += body.acc * DELTA_T/2

            # drift
            body.position += body.velocity * DELTA_T
        
        # update accelerations
        for body in self.bodies:
            self.acc_onBody(body)
                    
        for body in self.bodies:            
            # (1/2) kick
            body.velocity += body.acc * DELTA_T/2
        
        if self.surface:
            for body in self.bodies:
                body.draw(self.surface)