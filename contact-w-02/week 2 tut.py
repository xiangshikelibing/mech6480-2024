# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 00:47:22 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def fvm_solution(miu, dP_dx, num_cells, R):
    
    dr = R / num_cells
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

    A[0, 0] = 1
    A[0, 1] = -1
    b[0] = 0

    A[-1, -1] = 1
    b[-1] = 0

    u_fvm = np.linalg.solve(A, b)
    return r, u_fvm

def analytical_solution(x, u, t):
    
    
    rho_0 = lambda x: np.sin(np.pi * x)
    return rho_0(x - u * t)

if __name__ == "__main__":
    
    miu = 0.001  
    dP_dx = -0.001  
    R = 1.0  
    num_cells = 8  

    # Solve for FVM solution
    r, u_fvm = fvm_solution(miu, dP_dx, num_cells, R)

    # Analytical solution 
    u_const = 1.0  
    time = 1.0  

    x_analytical = np.linspace(0, R, 100)
    rho_analytical = analytical_solution(x_analytical, u_const, time)

    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, u_fvm, 'bo-', label='FVM Solution (Numerical)')
    ax.plot(x_analytical, rho_analytical, 'r-', label='Analytical Solution')
    ax.set_xlabel('Radial Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Comparison of FVM and Analytical Solutions')
    ax.grid(True)
    ax.legend()

    # timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)

   
    
    plt.show()