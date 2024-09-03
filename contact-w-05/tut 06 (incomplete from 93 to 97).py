# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:46:22 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt
import time as timer

#system parameters
LX = 1.0  # Channel length (4 times height)
LY = 0.25  # Channel height

RHO = 1.
MU = 0.01
nu = MU/RHO
g = 0.1 


#DISCRETISATION

NX = 20
NY = 20
DT = 0.01
NUM_STEPS = 1000
PLOT_EVERY = 100
N_PRESSURE_ITE = 100

# ALLOCATE MEMORY OF COMPUTE VARIABLES
u = np.ones((NX + 1, NY + 2), float)
v = np.zeros((NX + 2, NY + 1), float)
p = np.zeros((NX + 2, NY + 2), float)

ut = np.zeros((NX + 1, NY + 2), float)
vt = np.zeros((NX + 2, NY + 1), float)

prhs = np.zeros((NX + 2, NY + 2), float)

uu = np.zeros((NX + 1, NY + 1),float)
vv = np.zeros_like(uu)

xnodes = np.linspace(0, LX, NX+1)
ynodes = np.linspace(0, LY, NY+1)

J_u_x = np.zeros((NX,NY))
J_u_y = np.zeros((NX,NY+1))

J_v_x = np.zeros((NX+1,NY-1))
J_v_y = np.zeros((NX,NY))

dx = LX/NX
dy = LY/NY
dxdy = dx*dy

fig, ax1 = plt.subplots(1,1,figsize=[10,6])

def boundary_xvel(vel_field):
    vel_field[:,0] = - u[:,1]
    vel_field[:,-1] = - u[:,-2]
    return vel_field

def boundary_yvel(vel_field):
    vel_field[:,0] = 0. 
    vel_field[:,-1] = 0.
    return vel_field

u = boundary_xvel(u)
v = boundary_yvel(v)


time = 0
tic = timer.time()

for steps in range(NUM_STEPS):
    # Calculate fluxes
    J_u_x = 0.25 * (u[:-1, 1:-1] + u[1:, 1:-1])**2
    J_u_x -= nu * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx

    J_u_y = 0.25 * (u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1, :] + v[1:-2, :])
    J_u_y -= nu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy

    J_v_x = 0.25 * (u[:, 2:-1] + u[:, 1:-2]) * (v[1:, 1:-1] + v[:-1, 1:-1])
    J_v_x -= nu * (v[1:, 1:-1] - v[:-1, 1:-1]) / dx

    J_v_y = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1])**2
    J_v_y -= nu * (v[1:-1, 1:] - v[1:-1, :-1]) / dy

    ut[-1, 1:-1] = ut[0, 1:-1]
    
    
    ut = boundary_xvel(ut)ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (DT / dxdy) * (
        dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[1:-1, 1:] - J_u_y[1:-1, :-1])
    ) + DT * g
    vt = boundary_yvel(vt)

    #step 2
    
    divergence = (ut[1:,1:-1]-ut[:-1,1:-1])/dx + (vt[1:-1,1:]-vt[1:-1,:-1])/dy
    prhs = divergence * RHO / DT
    p_next = np.zeros_like(p)
    for _ in range (N_PRESSURE_ITE):
        p_next [1:-1,1:-1] = (-prhs*dxdy**2 + dy**2*(p[:-2,1:-1] + p[2:,1:-1]) + dx**2*(p[1:-1,:-2] +p[1:-1,2:]))/(2*dx**2 + 2*dy**2)
       
        p_next [0,:] = p_next[-2,:]
        p_next [-1,:] = -p_next[1,:]
        p_next [:,0] = p_next[:,1]
        p_next [:,-1] = p_next[:,-2]
        p = p_next.copy()
    
    #step 3
    
    u[1:-1,1:-1] = ut[1:-1,1:-1] - DT*(1./dx)*(p[2:-1,1:-1]-p[1:-2,1:-1])/RHO
    v[1:-1,1:-1] = vt[1:-1,1:-1] - DT*(1./dy)*(p[1:-1,2:-1]-p[1:-1,1:-2])/RHO
    
    u = boundary_xvel(u)
    v = boundary_yvel(v)

    time = time+DT
    if((steps+1) %PLOT_EVERY ==0):
        divu = (u[1:,1:-1]-u[:-1,1:-1])/dx + (v[1:-1,1:]-v[1:-1,:-1])/dy
        toc = timer.time()
        print(f"Step {steps+1},norm of div(u):{np.linalg.norm(divu):.4e}. \n Sec per it ={(toc-tic)/(steps+1):.4e}")
        
        uu = 0.5*(u[0:NX+1,1:NY+2] +u[0:NX+1,0:NY+1])
        vv = 0.5*(v[1:NX+2,0:NY+1] +v[0:NX+1,0:NY+1])
        xx, yy = np.meshgrid(xnodes, ynodes, indexing = 'ij')
        ax1.clear()
        ax1.contourf(xx, yy, np.sqrt(uu**2 + vv**2))
        ax1.quiver(xx, yy, uu, vv)
        ax1.plot(LX/2. + uu[NX//2,:], ynodes, 'k-', lw=3)
        if steps > 100:
            ax1.plot(LX/2. + uuprev, ynodes, 'r--', lw=2)
            
        uuprev = uu[NX//2,:]
        fig.savefig(f'channel_{(steps+1):4d}')
        plt.pause(0.1)
plt.show()