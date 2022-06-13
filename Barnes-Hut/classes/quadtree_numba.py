import numpy as np
from math import sqrt
from numba import deferred_type, optional,njit
from numba.experimental import jitclass
import numba
import pygame
DELTA_T = 0.0005
G = 6.674 * 10**(-11)

@njit
def _center_of_mass(positions,masses):
    totalMass = np.sum(masses)
    x = positions[:,0]
    y = positions[:,1]
    
    cx = np.sum(x*masses)/totalMass
    cy = np.sum(y*masses)/totalMass
    
    centerOfMass = np.array([cx,cy],dtype='float64')

    return centerOfMass, totalMass

@njit
def _compute_acceleration(r,d,bodyRadius,otherBodyMass):
    if r<=2*bodyRadius:
        SOFTENING = 2*bodyRadius
    else:
        SOFTENING = 0
    return -G * otherBodyMass * d * (r+SOFTENING)**(-3)

@njit
def points_distance(posA,posB):
    displacement = posA - posB
    distance = sqrt((displacement[0])**2+(displacement[1])**2)
    return distance, displacement

@njit
def are_points_equal(posA,posB,tol=1e-16):
    if abs(posA[0] - posB[0]) < tol and abs(posA[1] - posB[1]) < tol:
        return True
    else:
        return False    

@njit
def boundary_contains_point(cx,cy,side,posP):
    left, right = cx - side/2, cx + side/2
    bottom, top = cy - side/2, cy + side/2
    if posP[0] >= left and posP[0] <= right and posP[1]<=top and posP[1]>=bottom:
        return True
    else:
        return False   

QuadTree_type = deferred_type()
spec = {
    'cx' : numba.float64,
    'cy' : numba.float64,
    'side' : numba.float64,
    'max_points' : numba.uint8,
    'points_pos' :  numba.float64[:,:],
    'points_masses' : numba.float64[:],
    'nPoints' : numba.uint8,
    'isSubdivided' : numba.boolean,
    'CM_pos' : numba.float64[:],
    'CM_mass' : numba.float64,
    'BotLeft' : optional(QuadTree_type),
    'TopLeft' : optional(QuadTree_type),
    'BotRight' : optional(QuadTree_type),
    'TopRight' : optional(QuadTree_type)
}
@jitclass(spec)
class QuadTree:
    '''
    TopLeft    TopRight
    
    BotLeft    BotRight
    '''
    def __init__(self,cx,cy,side,max_points):
        self.cx = cx
        self.cy = cy
        self.side = side
        self.max_points = max_points

        self.points_pos = np.zeros((max_points,2),dtype='float64')
        self.points_masses = np.zeros(max_points,dtype='float64')
        self.nPoints = 0

        self.isSubdivided = False
        
        self.BotLeft, self.TopLeft, self.BotRight, self.TopRight = None, None, None, None
        
        self.CM_pos = np.zeros(2,dtype='float64')
        self.CM_mass = 0
    def update_points(self,posP,massP):
        self.points_pos[self.nPoints] = posP
        self.points_masses[self.nPoints] = massP
        self.nPoints += 1
    def update_CM(self,posCM,massCM):
        self.CM_pos = posCM
        self.CM_mass = massCM
    def subdivide(self):
        side = self.side / 2
        
        cx,cy = self.cx-side/2,self.cy-side/2
        self.BotLeft = QuadTree(cx,cy,side,self.max_points)
        cx,cy = self.cx-side/2,self.cy+side/2
        self.TopLeft = QuadTree(cx,cy,side,self.max_points)
        cx,cy = self.cx+side/2,self.cy-side/2
        self.BotRight = QuadTree(cx,cy,side,self.max_points)
        cx,cy = self.cx+side/2,self.cy+side/2
        self.TopRight = QuadTree(cx,cy,side,self.max_points)
        
        self.isSubdivided = True

        posToBePushed = self.points_pos.copy()
        massToBePushed = self.points_masses.copy()
        nPoints = self.nPoints

        self.points_pos = np.zeros((self.max_points,2),dtype='float64')
        self.points_masses = np.zeros(self.max_points,dtype='float64')
        self.nPoints = 0

        for i in range(nPoints):
            if boundary_contains_point(self.BotLeft.cx,self.BotLeft.cy,self.BotLeft.side,posToBePushed[i]):
                self.BotLeft.update_points(posToBePushed[i],massToBePushed[i])
            elif boundary_contains_point(self.TopLeft.cx,self.TopLeft.cy,self.TopLeft.side,posToBePushed[i]):
                self.TopLeft.update_points(posToBePushed[i],massToBePushed[i])
            elif boundary_contains_point(self.BotRight.cx,self.BotRight.cy,self.BotRight.side,posToBePushed[i]):
                self.BotRight.update_points(posToBePushed[i],massToBePushed[i])
            elif boundary_contains_point(self.TopRight.cx,self.TopRight.cy,self.TopRight.side,posToBePushed[i]):
                self.TopRight.update_points(posToBePushed[i],massToBePushed[i])
            
QuadTree_type.define(QuadTree.class_type.instance_type)               

@njit
def insert(tree,posP,massP):
    if not boundary_contains_point(tree.cx,tree.cy,tree.side,posP): #point not inside the boundary
        return False
    else:
        if tree.nPoints < tree.max_points: #quad free
            if not tree.isSubdivided:
                tree.update_points(posP,massP)
                return True
            else:
                return insertToChildren(tree,posP,massP)
        else:
            if tree != None and not tree.isSubdivided:
                tree.subdivide()
                return insertToChildren(tree,posP,massP)
            else:
                return insertToChildren(tree,posP,massP)
    
@njit               
def insertToChildren(tree,posP,massP):
    return (insert(tree.BotLeft,posP,massP) or insert(tree.TopLeft,posP,massP) or insert(tree.BotRight,posP,massP) or insert(tree.TopRight,posP,massP))

@njit 
def centerOfMass(tree):
    if tree.isSubdivided:
        centerOfMass(tree.BotLeft)
        centerOfMass(tree.TopLeft)
        centerOfMass(tree.BotRight)
        centerOfMass(tree.TopRight)
        
        vet_pos = np.zeros((4,2),dtype='float64')
        vet_pos[0], vet_pos[1], vet_pos[2], vet_pos[3] = tree.BotLeft.CM_pos,tree.TopLeft.CM_pos,tree.BotRight.CM_pos,tree.TopRight.CM_pos
        vet_mas = np.array([tree.BotLeft.CM_mass,tree.TopLeft.CM_mass,tree.BotRight.CM_mass,tree.TopRight.CM_mass],dtype='float64')
        
        CM_pos, CM_mass = _center_of_mass(vet_pos,vet_mas)
        tree.update_CM(CM_pos,CM_mass)
    else:
        if tree.nPoints>0:
            CM_pos, CM_mass = _center_of_mass(tree.points_pos[:tree.nPoints],tree.points_masses[:tree.nPoints])
            tree.update_CM(CM_pos,CM_mass)
        else:
            tree.update_CM(np.zeros(2,dtype='float64'),0)

@njit
def accOfTree_onBody(tree, posP, theta, acc):
    if are_points_equal(tree.CM_pos, posP) or tree.CM_mass == 0:
        return acc
        
    distance, displacement = points_distance(posP,tree.CM_pos) #distance between P and the center of mass of the node
    cellSize = tree.side #size of the node-cell
    if cellSize/distance < theta or not tree.isSubdivided:
        a = _compute_acceleration(distance,displacement,2, tree.CM_mass)
        acc += a
    else:
        acc = accOfTree_onBody(tree.BotLeft,posP, theta, acc)
        acc = accOfTree_onBody(tree.TopLeft, posP, theta, acc)
        acc = accOfTree_onBody(tree.BotRight, posP, theta, acc)
        acc = accOfTree_onBody(tree.TopRight, posP, theta, acc)

    return acc

@njit
def computeNewFrame(position, mass, velocity, acceleration,cx,cy,side,max_points,theta):
    #The positions and velocities are updated using a leap-frog scheme
    # First (1/2) kick
    velocity += acceleration * DELTA_T/2

    # drift
    position += velocity * DELTA_T
    
    #build quadtree given updated positions
    nBodies = len(position)
    tree = QuadTree(cx,cy,side,max_points)
    for i in range(nBodies):
        insert(tree,position[i],mass[i])
    
    #compute center of mass
    centerOfMass(tree)

    # update accelerations
    for i in range(nBodies):
        acceleration[i] = accOfTree_onBody(tree,position[i],theta,np.zeros(2,dtype='float64'))
                
    # Last (1/2) kick
    velocity += acceleration * DELTA_T/2
    return position, velocity

class System_Quadtree_Numba:
    def __init__(self,position, mass, velocity, acceleration, color, cx, cy, side, max_points, theta, surface):
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
        self.surface = surface

    def computeNewFrame(self):
        self.position, self.velocity = computeNewFrame(self.position, self.mass, self.velocity, self.acceleration,self.cx,self.cy,self.side,self.max_points,self.theta)
        
        if self.surface:
            for i in range(len(self.position)):
                pygame.draw.circle(self.surface, self.color[i], (self.position[i][0] , self.surface.get_height() - self.position[i][1]), 1)