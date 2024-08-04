# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 00:10:27 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from git import Repo

import os

git_path = "/path/to/git"  # Replace with the actual path to your Git executable
os.environ["PATH"] += os.pathsep + git_path

def poiseuille_flow(miu, dP_dx, dr, num_cells, R):
    
    # Radial positions 
    r = np.linspace(0.5 * dr, R - 0.5 * dr, num_cells)

    A = np.zeros((num_cells, num_cells))
    b = np.zeros(num_cells)

    for i in range(1, num_cells - 1):
        r_N = r[i] + 0.5 * dr
        r_S = r[i] - 0.5 * dr
        A[i, i - 1] = miu * r_S / dr
        A[i, i] = -miu * (r_N + r_S) / dr
        A[i, i + 1] = miu * r_N / dr
        b[i] = dP_dx * r[i] * dr

    # Boundary conditions, symmetric
    A[0, 0] = 1
    A[0, 1] = -1
    b[0] = 0

    A[-1, -1] = 1
    b[-1] = 0

    return A, b, r

def solve_velocity_profile(miu, dP_dx, num_cells, R):
    dr = R / num_cells
    A, b, r = poiseuille_flow(miu, dP_dx, dr, num_cells, R)
    u = np.linalg.solve(A, b)
    return r, u

if __name__ == "__main__":
    
    miu = 0.001  
    dP_dx = -0.001  
    R = 1.0  
    num_cells = 8  

    
    r, u = solve_velocity_profile(miu, dP_dx, num_cells, R)

    
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, u, 'bo-', label='Numerical Solution')
    plt.xlabel('Radial Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    #timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)
    
    #ID
    try:
        repo = Repo('.', search_parent_directories=True)
        revsha = repo.head.object.hexsha[:8]
        ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
    except Exception as e:
        ax.annotate("[rev unknown]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
        print(f"Error accessing Git repository: {e}")

    
    