# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:06:27 2024

@author: xiangshikelibing
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time

def steady_diffusion(k:float, dx, num_cells, T_A, T_B, q)->float:
    # internal notes
    a_E = k / dx
    a_W = k / dx
    a_P = a_E + a_W 
    b_P = -q * dx
    # boundary conditions
    S_u = -2.0 * k / dx
    a_P_star = 3.0 * k / dx
    
    # Set up system of equations
    A = (np.diag(np.repeat(a_W,num_cells-1),-1) + 
         np.diag(np.repeat(-a_P,num_cells),0) + 
         np.diag(np.repeat(a_E,num_cells-1),1))
    
    #overwrite boundary row
    A[0,0] = -a_P_star
    A[-1,-1] = -a_P_star
    #set up RHS
    b = np.ones(num_cells) * b_P
    b[0] += S_u * T_A
    b[-1] += S_u * T_B
    
    return A, b.transpose()

if __name__ == "__main__":
    # problem setting
    k = 0.5       # W/m.K
    length = 0.02 # m
    T_A = 100     # K or C
    T_B = 200     # K or C
    q = 1000e3    # kW/m^3
    
    # discretisation
    num_cells = 5
    dx = length / num_cells
    x = np.linspace(0.5 * dx, (num_cells-0.5) * dx, num_cells)
    
    A, b =steady_diffusion(k, dx, num_cells, T_A, T_B, q)
    T = np.linalg.solve(A,b)
    plt.plot(x, T, 'bo')
    # calculate analytic solution:
    def analytic_solution(x):
        return ((T_B - T_A) / length + q * (length - x) / (2 * k)) * x + T_A
    x_a = np.linspace (0, length, 100)
    soln = analytic_solution(x_a)
    #plt.plot([0,0.5], [T_A, T_B], 'r--')