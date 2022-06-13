import numpy as np
from main_simulation import generateBodies
from classes.quadtree_numba import System_Quadtree_Numba
from classes.quadtree import System_Quadtree
from classes.system_naive import System_Naive
import time
import matplotlib.pyplot as plt

def exe(mode, n_iters, n_galaxies, n_rings, n_particles_per_ring):

    position, mass, velocity, acceleration, color = generateBodies(n_galaxies, n_rings, n_particles_per_ring)
    cx, cy, side, max_points, theta = 500, 500, 1000, 1, 1

    if mode == 0:
        system = System_Quadtree_Numba(position, mass, velocity, acceleration, color, cx, cy, side, max_points, theta, None)
    elif mode == 1:
        system = System_Quadtree(position, mass, velocity, color, cx, cy, side, max_points, theta, None)
    elif mode == 2:
        system = System_Naive(position, mass, velocity, color, None)

    
    i = 0
    iters_exe = []
    while i<n_iters:
        start = time.perf_counter()
        system.computeNewFrame()
        end = time.perf_counter()
        iters_exe.append(end-start)
        i+=1
    
    return iters_exe[1:]

if __name__ == "__main__":
    # 0: Quadtree_Numba, 1: Quadtree, 2: Naive
    n_iters = 100
    vet_n_bodies = range(10,250,3)
    for MODE in [0,1]:
        ris = []
        for n_bodies in vet_n_bodies:
            n_galaxies = 2
            n_bodies_per_galaxy = n_bodies/n_galaxies
            n_rings = 3
            n_particles_per_ring = int(n_bodies_per_galaxy/n_rings)

            iters_exe = exe(MODE, n_iters, n_galaxies, n_rings, n_particles_per_ring)

            ris.append([n_bodies,np.mean(iters_exe)])
            print('n bodies ',n_bodies,' mean exe time ',np.mean(iters_exe),' min ',np.min(iters_exe),' max ',np.max(iters_exe))
        ris = np.array(ris)
        plt.plot(ris[:,0],ris[:,1],label=MODE)
    plt.legend()
    plt.show()
