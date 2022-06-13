import numpy as np
from classes.simulation import Simulation
G = 6.674 * 10**(-11)

#generate N particles in a circular orbit around the BlackHole
def particle_ring(N,index, radius, BH_position, BH_mass, BH_velocity, galaxy_color, position, velocity, mass, color):
    theta = 0
    arclen = (2 * np.pi) / N
    v = np.sqrt(G * BH_mass / radius)
    i = 0
    while i < N:
        angle = theta * arclen
        beta = angle + np.pi / 2
        theta += 1

        position[index + i] = BH_position + (radius * np.cos(angle), radius * np.sin(angle))
        velocity[index + i] =  BH_velocity + (v * np.cos(beta), v * np.sin(beta))
        mass[index + i] =  6 * pow(10, 10)
        color[index + i] = galaxy_color
        i+=1 

    return position, mass, velocity, color, index + i

def generateBodies(n_galaxies,n_rings, n_particles_per_ring):
    n = n_galaxies*(n_rings * n_particles_per_ring + 1)

    position = np.zeros((n,2),dtype='float64')
    mass = np.zeros(n,dtype='float64')
    velocity = np.zeros((n,2),dtype='float64')
    acceleration = np.zeros((n,2),dtype='float64')
    color = np.zeros((n,3),dtype='int')

    i = 0
    #GALAXY 1
    position[i], mass[i], velocity[i], color[i] = (300,300),  6 * pow(10, 16), (20,0), (255,255,255) #BlackHole
    BH_position, BH_mass, BH_velocity = position[i], mass[i], velocity[i]
    i+=1
    rings = np.linspace(20, 150, n_rings)
    particles = np.full_like(rings, n_particles_per_ring)
    for j in range(len(particles)):
        position, mass, velocity, color, i = particle_ring(particles[j], i, rings[j], BH_position, BH_mass, BH_velocity, (169,247,255), position, velocity, mass, color)


    #GALAXY 2
    position[i], mass[i], velocity[i], color[i] = (700,700),  6 * pow(10, 16), (-20,0), (255,255,255) #BlackHole
    BH_position, BH_mass, BH_velocity = position[i], mass[i], velocity[i]
    i+=1
    rings = np.linspace(20, 150, n_rings)
    particles = np.full_like(rings, n_particles_per_ring)
    for j in range(len(particles)):
        position, mass, velocity, color, i = particle_ring(particles[j], i, rings[j], BH_position, BH_mass, BH_velocity, (255,169,247),  position, velocity, mass, color)
    
    return  position, mass, velocity, acceleration, color

if __name__ == "__main__": 
    # 0: Quadtree_Numba, 1: Quadtree, 2: Naive
    MODE = 0
    n_galaxies, n_rings, n_particles_per_ring = 2, 10, 50
    cx, cy, side, max_points, theta = 500, 500, 1000, 1, 1
    saveFrame = True 
    position, mass, velocity, acceleration, color = generateBodies(n_galaxies, n_rings, n_particles_per_ring)
    sim = Simulation(
        position, mass, velocity, acceleration, color,
        cx, cy, side, max_points, theta,
        MODE, saveFrame)