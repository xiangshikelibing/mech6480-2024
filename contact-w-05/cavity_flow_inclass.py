"""
MECH6480 - WEEK 5 - Cavity Flow Example
We will develop this code in the Contact session on Wednesday.

Problem description:
 Re = 100
 rho = 1
 nu = 0.01
 Lx = Ly = 1
 
 u = Re*nu/Lx
   = 1e2*1e-2/1
   = 1
 
 Problem domain:
        lid: u = 1, v = 0
        +->->->->->->->->->->->->->-+
        |                           |
        |                           |
        |                           |
  wall  |                           |  wall
 u = 0  |                           |  u = 0
 v = 0  |                           |  v = 0
        |                           |
        |                           |
        +---------------------------+
        wall: u = 0, v = 0


 The staggered grid with ghost cells: 

    •   →   •   →   •   →   •   →   •
        |       |       |       |    
    ↑ - +---↑---+---↑---+---↑---+ - ↑
        :       |       |       :    
    •   →   •   →   •   →   •   →   •
        :       |       |       :    
    ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑
        :       |       |       :    
    •   →   •   →   •   →   •   →   •
        :       |       |       :    
    ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑
        :       |       |       :    
    •   →   •   →   •   →   •   →   •
        :       |       |       :    
    ↑ - 0---↑---+---↑---+---↑---+ - ↑
        |       |       |       |    
    •   →   •   →   •   →   •   →   •

 • Pressure stored at the cell centers
 → Horizontal velocity stored at the cell faces
 ↑ Vertical velocity stored at the cell faces
 0 Indicates origin of the grid
        
"""
import numpy as np
import matplotlib.nyplot as plt
import time as timer

#system parameters
LX = 1.
LY = 1.

RHO = 1.
MU = 0.01
nu = MU/RHO

#BOUNDARY
UNORTH = 1.
USOUTH = 0.
VEAST = 0.
VWEST = 0.

#DISCRETISATION

NX = 20
NY = 20
DT = 0.01
NUM_STEPS = 1000
PLOT_EVERY = 100

# ALLOCATE MEMORY OF COMPUTE VARIABLES
u = np.zeros((NX + 1, NY + 2), float)
v = np.zeros((NX + 2, NY + 1), float)
p = np.zeros((NX + 2, NY + 2), float)

ut = np.zeros((NX + 1, NY + 2), float)
vt = np.zeros((NX + 2, NY + 1), float)

prhs = np.zeros((NX + 2, NY + 2), float)

uu = np.zeros((NX + 1, NY + 1),float)
vv = np.zeros_like(uu)

xnodes = np.linspace(0, LX, NX+1)
ynodes = np.linspace(0, LY, NY+1)

J_u_x = np.zerps((NX,NY))
J_u_y = np.zerps((NX-1,NY+1))

J_v_x = np.zerps((NX+1,NY-1))
J_v_y = np.zerps((NX,NY))

dx = LX/NX
dy = LY/NY
dxdy = dx*dy

#fig.ax1 = plt.subplots(1,1,figsize=[6,6])

time = 0
tic = timer.time()

u[:,0] = 2.* USOUTH - u[:,1]
u[:,-1] = 2.* UNORTH - u[:,-2]
v[0,:] = 2.* VWEST - v[1,:]
v[-1,:] = 2.* VEAST - v[-2,:]

for steps in range(NUM_STEPS): 
    J_u_x = 0.25 *(u[:-1, 1:-1] + u[1:, 1:-1])**2
    J_u_x -= nu*(u[1:, 1:-1] - u[:-1, 1:-1])/dx
    
    J_u_y = 0.25 *(u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1,:] + v[1:-2,:])
    J_u_y -= nu*(u[1:-1,1:]-u[1:-1,:-1])/dy
    
    J_v_x = 0.25 *(u[:,2:-1]+u[:,1:-2]) * (v[1:,1:-1] +v[:-1,1:-1])
    J_v_x -= nu*(v[1:,1:-1]-v[:-1,1:-1])/dx
    
    J_v_y = 0.25 *(v[1:-1,1:]+v[1:-1,:-1])**2 
    J_v_y -= nu*(v[1:-1,1:]-v[1:-1,:-1])/dy

    ut[1:-1,1:-1] = u[1:-1,1:-1] - (DT/dxdy) * (dy * (J_u_x[1:,:]-J_u_x[:-1,:])+ dx*(J_u_y[:,1:]-J_u_y[:,:-1]))
    vt[1:-1,1:-1] = v[1:-1,1:-1] - (DT/dxdy) * (dy * (J_v_x[1:,:]-J_v_x[:-1,:])+ dx*(J_v_y[:,1:]-J_v_y[:,:-1]))

    u[:,0] = 2.* USOUTH - u[:,1]
    u[:,-1] = 2.* UNORTH - u[:,-2]
    v[0,:] = 2.* VWEST - v[1,:]
    v[-1,:] = 2.* VEAST - v[-2,:]
    
    divergence = (ut[1:,1:-1]-ut[:-1,1:-1])/dx + (vt[1:-1,1:]-vt[1:-1,:-1])/dy
    prhs = divergence * RHO / DT
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    



