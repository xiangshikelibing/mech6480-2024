# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:59:15 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Domain length
H = 1.0  # Domain height
Re_values = [1000, 2500]  # Reynolds numbers to simulate
rho = 1.0  # Density (kg/m^3)
U_lid = 1.0  # Lid velocity (m/s)

# Discretization
N = 100  # Number of grid points
dx = L / (N - 1)
dy = H / (N - 1)
dt = 0.001  # Time step
max_iter = 10000  # Maximum number of time steps
tolerance = 1e-6  # Tolerance for steady-state

# Grids
x = np.linspace(0, L, N)
y = np.linspace(0, H, N)
X, Y = np.meshgrid(x, y)

# Initialize fields (u, v for velocity; p for pressure)
def initialize_fields():
    u = np.zeros((N, N))  # Velocity in x-direction
    v = np.zeros((N, N))  # Velocity in y-direction
    p = np.zeros((N, N))  # Pressure field
    return u, v, p

# Boundary conditions
def apply_boundary_conditions(u, v):
    u[0, :] = 0  # Left wall
    u[-1, :] = 0  # Right wall
    u[:, 0] = 0  # Bottom wall
    u[:, -1] = U_lid  # Lid velocity
    
    v[0, :] = 0  # Left wall
    v[-1, :] = 0  # Right wall
    v[:, 0] = 0  # Bottom wall
    v[:, -1] = 0  # Lid

# Pressure-Poisson solver
def pressure_poisson(p, b, dx, dy):
    pn = np.copy(p)
    for _ in range(50):  # Perform 50 iterations for the Poisson equation
        pn = np.copy(p)
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        
        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dy = 0 at the top (lid)
        p[:, 0] = p[:, 1]  # dp/dy = 0 at the bottom
        p[-1, :] = p[-2, :]  # dp/dx = 0 at the right wall
        p[0, :] = p[1, :]  # dp/dx = 0 at the left wall
    return p

# Build the RHS of the pressure Poisson equation
def build_rhs(u, v, dx, dy, dt):
    b = np.zeros((N, N))
    b[1:-1, 1:-1] = (rho * (1/dt * 
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) + 
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx))**2 -
                     2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy) * 
                          (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy))**2))
    return b

# Velocity update function
def update_velocity(u, v, p, dx, dy, dt, nu):
    un = np.copy(u)
    vn = np.copy(v)
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
    
    # Apply boundary conditions after velocity update
    apply_boundary_conditions(u, v)

# Main solver function
def lid_driven_cavity_flow(Re):
    nu = U_lid * L / Re  # Kinematic viscosity
    
    # Initialize velocity and pressure fields
    u, v, p = initialize_fields()
    
    apply_boundary_conditions(u, v)
    
    for n in range(max_iter):
        un = np.copy(u)
        vn = np.copy(v)
        
        b = build_rhs(u, v, dx, dy, dt)
        p = pressure_poisson(p, b, dx, dy)
        update_velocity(u, v, p, dx, dy, dt, nu)
        
        # Check for steady-state condition (when velocity change is negligible)
        if np.linalg.norm(u - un) < tolerance and np.linalg.norm(v - vn) < tolerance:
            print(f"Steady-state reached after {n} iterations")
            break
    
    return u, v, p

# Plotting functions
def plot_velocity_profiles(u, v, Re):
    mid_x = int(N/2)
    mid_y = int(N/2)
    
    # Vertical profile at x = L/2
    plt.figure()
    plt.plot(u[:, mid_x], y, label='Numerical')
    plt.xlabel('Vertical Velocity (u) at x=L/2')
    plt.ylabel('y')
    plt.title(f'Vertical Velocity Profile for Re = {Re}')
    plt.grid(True)
    plt.show()

    # Horizontal profile at y = H/2
    plt.figure()
    plt.plot(x, v[mid_y, :], label='Numerical')
    plt.xlabel('x')
    plt.ylabel('Horizontal Velocity (v) at y=H/2')
    plt.title(f'Horizontal Velocity Profile for Re = {Re}')
    plt.grid(True)
    plt.show()

def plot_streamlines(u, v, Re):
    plt.figure()
    plt.streamplot(X, Y, u, v, density=2, linewidth=1)
    plt.title(f'Streamlines for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Run the solver and generate plots for both Re = 1000 and Re = 2500
for Re in Re_values:
    u, v, p = lid_driven_cavity_flow(Re)
    plot_velocity_profiles(u, v, Re)
    plot_streamlines(u, v, Re)
