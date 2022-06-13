import numpy as np
from classes.classes import Body,Square
from classes.quadtree_numba import _compute_acceleration

DELTA_T = 0.0005
G = 6.674 * 10**(-11)

def _center_of_mass(points):
    vet = np.array([[P.position[0],P.position[1],P.mass] for P in points])
    centerOfMass = np.average(vet[:,:2], axis=0, weights=vet[:,2])
    totalMass = sum(vet[:,2])
    return Body(centerOfMass[0],centerOfMass[1],totalMass,radius=10)
             
class QuadTree:
    '''
    TopLeft    TopRight
    
    BotLeft    BotRight
    '''
    def __init__(self,boundary,max_points):
        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.isSubdivided = False
        self.BotLeft, self.TopLeft, self.BotRight, self.TopRight = None, None, None, None
        self.CM = Body(0,0,0)
                
    def insert(self,P):       
        if not self.boundary.contains(P): #point not inside the boundary
            return False
        
        if (len(self.points) < self.max_points): #quad free
            if not self.isSubdivided:
                self.points.append(P)
                return True
        
        if not self.isSubdivided:
            self.subdivide()
     
        return self.insertToChildren(P)
        
    def subdivide(self):
        side = self.boundary.side / 2

        self.BotLeft = QuadTree(Square(self.boundary.cx-side/2,self.boundary.cy-side/2,side),self.max_points)
        self.TopLeft = QuadTree(Square(self.boundary.cx-side/2,self.boundary.cy+side/2,side),self.max_points)
        self.BotRight = QuadTree(Square(self.boundary.cx+side/2,self.boundary.cy-side/2,side),self.max_points)
        self.TopRight = QuadTree(Square(self.boundary.cx+side/2,self.boundary.cy+side/2,side),self.max_points)
        
        self.isSubdivided = True
        
        pointsToBePushed = self.points.copy()
        self.points = []
        for P in pointsToBePushed:
            self.insertToChildren(P)
                   
    def insertToChildren(self,P):
        return (self.BotLeft.insert(P) or self.TopLeft.insert(P) or self.BotRight.insert(P) or self.TopRight.insert(P))
    
    def centerOfMass(self):
        if self.isSubdivided:
            self.BotLeft.centerOfMass()
            self.TopLeft.centerOfMass()
            self.BotRight.centerOfMass()
            self.TopRight.centerOfMass()
            self.CM = _center_of_mass([self.BotLeft.CM,self.TopLeft.CM,self.BotRight.CM,self.TopRight.CM])
        else:
            if len(self.points)>0:
                self.CM = _center_of_mass(self.points)
    
    def accOfTree_onBody(self, P, theta, acc):
        if self.CM.isEqualTo(P) or self.CM.mass == 0: #check that cell is not empty and not coincide with P
            return acc
                   
        distance, displacement = P.distance(self.CM) #distance between P and the center of mass of the node
        cellSize = self.boundary.side #size of the node-cell
        if cellSize/distance < theta or not self.isSubdivided:
            a = _compute_acceleration(distance,displacement,P.radius, self.CM.mass)
            acc += a
        else:
            acc = self.BotLeft.accOfTree_onBody(P, theta, acc)
            acc = self.TopLeft.accOfTree_onBody(P, theta, acc)
            acc = self.BotRight.accOfTree_onBody(P, theta, acc)
            acc = self.TopRight.accOfTree_onBody(P, theta, acc)

        return acc

           
def computeNewFrame(bodies,boundary,max_points,theta):
    #The positions and velocities are updated using a leap-frog scheme
    for body in bodies:
        # (1/2) kick
        body.velocity += body.acc * DELTA_T/2

        # drift
        body.position += body.velocity * DELTA_T
    
    #build quadtree given updated positions
    tree = QuadTree(boundary,max_points)
    for body in bodies:
        tree.insert(body)
    
    #compute center of mass
    tree.centerOfMass()

    # update accelerations
    for body in bodies:
        body.acc = tree.accOfTree_onBody(body,theta,np.zeros(2))
                
    for body in bodies:            
        # (1/2) kick
        body.velocity += body.acc * DELTA_T/2

    return bodies

class System_Quadtree:
    def __init__(self, position, mass, velocity, color, cx, cy, side, max_points, theta, surface):
        self.bodies = self.initialize_bodies(position, mass, velocity, color)
        self.boundary = Square(cx,cy,side)

        self.max_points = max_points
        self.theta = theta
        self.surface = surface

    def initialize_bodies(self, position, mass, velocity, color):
        bodies = []
        for i in range(len(position)):
            body = Body(position[i][0],position[i][1],mass[i],velocity[i],color[i],1)
            bodies.append(body)
        return bodies

    def computeNewFrame(self):
        self.bodies = computeNewFrame(self.bodies,self.boundary,self.max_points,self.theta)

        if self.surface:
            for body in self.bodies:
                body.draw(self.surface)