# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:48:05 2024

@author: 13747
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from git import Repo

# Parameters
LX = 10 # m
LY = 10 # m
ALPHA = 4 # W/mK (thermal diffusivity)
S = 500 

# Grid
NX = 21
NY = 21
dx = LX / NX
dy = LY / NY
x = np.linspace(dx/2., LX-dx/2.0, NX) 
y = np.linspace(dy/2., LY-dy/2.0, NY)
xx, yy = np.meshgrid(x, y, indexing='ij') 
print(f"dx={dx:.4f}, dy={dy:.4f}")

# Time
simulated_time = 0
iteration = 0
T_END = 62.5e-3 
DT = 0.0000625
PLOT_EVERY = 100

# Initial and boundary conditions
T_INITIAL = 300.0
T_TOP = 350.0
T_LEFT = 400.0

T = np.ones((NX, NY)) * T_INITIAL

# Set boundary conditions
T[:, -1] = T_TOP   
T[0, :] = T_LEFT   

# Allocate memory for fluxes
x_flux = np.zeros((NX+1, NY))
y_flux = np.zeros((NX, NY+1))


def add_annotations(ax):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)
    
    try:
        repo = Repo('.', search_parent_directories=True)
        revsha = repo.head.object.hexsha[:8]
        ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
    except Exception as e:
        ax.annotate("[rev unknown]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
        print(f"Error accessing Git repository: {e}")

# Plot setup
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])  

tic = time.time()
while simulated_time < T_END:
    # Calculate fluxes - interior
    x_flux[1:-1,:] = ALPHA*(T[1:,:] - T[:-1,:])/dx
    y_flux[:,1:-1] = ALPHA*(T[:,1:] - T[:,:-1])/dy

    # Calculate fluxes 
    x_flux[0,:]  = ALPHA*(T[0,:] - T[1,:])/(dx/2.)  
    x_flux[-1,:] = 0  
    y_flux[:,0]  = ALPHA*(T[:,0] - T[:,1])/(dy/2.)  
    y_flux[:,-1] = 0  

    # Update T with S
    T = T + DT * (dy*(x_flux[1:,:] - x_flux[:-1,:]) \
                + dx*(y_flux[:,1:] - y_flux[:,:-1]))/(dx*dy) + S*DT

    # Update time    
    simulated_time += DT
    iteration += 1
    
    # Plotting
    if np.mod(iteration, PLOT_EVERY) == 0:
        ax0.cla()
        contour = ax0.contourf(xx, yy, T, vmin=T_INITIAL, vmax=T_LEFT, cmap='hot')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_title('Temperature ' + str(round(simulated_time,5)) + ' s')
        ax0.set_aspect('equal')
        add_annotations(ax0)  
        fig.colorbar(contour, ax=ax0)  
        fig.savefig(f'heatmap_{iteration:04d}.png')

# Plot final T distribution
ax0.cla()
contour = ax0.contourf(xx, yy, T, vmin=T_INITIAL, vmax=T_LEFT, cmap='hot')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_title('Temperature ' + str(round(simulated_time,5)) + ' s')
ax0.set_aspect('equal')
add_annotations(ax0)  
fig.colorbar(contour, ax=ax0)  
plt.show()

# Plot profile through the center
fig, ax = plt.subplots()
ax.plot(x, T[:, NY//2])
ax.set_xlabel('x')
ax.set_ylabel('T')
ax.set_title('Temperature profile y=L/2')
add_annotations(ax)  
plt.show()

# Final temperature field at steady state
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(xx, yy, T, vmin=T_INITIAL, vmax=T_LEFT, cmap='hot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Temperature field at steady state')
add_annotations(ax)  
fig.colorbar(contour, ax=ax)  
plt.show()

# Temperature profile along y = H/2 (horizontal line)
fig, ax = plt.subplots()
ax.plot(x, T[:, NY//2], label='y = H/2')
ax.set_xlabel('x')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature Profile along y = H/2')
ax.legend()
add_annotations(ax)  
plt.show()

# Temperature profile along x = L/2 (vertical line)
fig, ax = plt.subplots()
ax.plot(y, T[NX//2, :], label='x = L/2')
ax.set_xlabel('y')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature Profile along x = L/2')
ax.legend()
add_annotations(ax)  
plt.show()

print("Total elapsed time:", round(time.time()-tic,2), "s (", round((time.time()-tic)/60.0,2), "min)")
