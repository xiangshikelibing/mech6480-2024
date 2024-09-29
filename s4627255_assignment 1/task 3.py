# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:48:19 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  
from git import Repo  

L = 1.0  
H = 1.0  
Re_values = [1000, 2500]  
rho = 1.0  
U_lid = 1.0  

N = 100  
dx = L / (N - 1)
dy = H / (N - 1)
dt = 0.001  
max_iter = 10000  
tolerance = 1e-6  
save_interval = 100  

x = np.linspace(0, L, N)
y = np.linspace(0, H, N)
X, Y = np.meshgrid(x, y)

def add_annotations(ax):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='axes fraction', annotation_clip=False)
    
    try:
        repo = Repo('.', search_parent_directories=True)
        revsha = repo.head.object.hexsha[:8]
        ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='axes fraction', annotation_clip=False)
    except Exception as e:
        ax.annotate("[rev unknown]", xy=(0.05, 0.95), xycoords='axes fraction', annotation_clip=False)
        print(f"Error accessing Git repository: {e}")

def initialize_fields():
    u = np.zeros((N, N))  
    v = np.zeros((N, N))  
    p = np.zeros((N, N))  
    return u, v, p

def apply_boundary_conditions(u, v):
    u[0, :] = 0  
    u[-1, :] = 0  
    u[:, 0] = 0  
    u[:, -1] = U_lid  
    
    v[0, :] = 0  
    v[-1, :] = 0  
    v[:, 0] = 0  
    v[:, -1] = 0  

def pressure_poisson(p, b, dx, dy):
    pn = np.copy(p)
    for _ in range(50):  
        pn = np.copy(p)
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        
        p[:, -1] = p[:, -2]  
        p[:, 0] = p[:, 1]  
        p[-1, :] = p[-2, :]  
        p[0, :] = p[1, :]  
    return p

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
    
    apply_boundary_conditions(u, v)

def plot_velocity_magnitude_quiver(u, v, n):
    velocity_magnitude = np.sqrt(u**2 + v**2)
    
    fig, ax = plt.subplots()
    cont = ax.contourf(X, Y, velocity_magnitude, cmap='viridis')
    plt.colorbar(cont, ax=ax, label='Velocity Magnitude')
    
    ax.quiver(X[::5, ::5], Y[::5, ::5], u[::5, ::5], v[::5, ::5], scale=5)
    ax.set_title(f'Velocity Field at iteration {n}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    add_annotations(ax)
    plt.savefig(f'velocity_field_{n}.png')
    plt.close()

def lid_driven_cavity_flow_with_plotting(Re):
    nu = U_lid * L / Re  
    
    u, v, p = initialize_fields()
    
    apply_boundary_conditions(u, v)
    
    for n in range(max_iter):
        un = np.copy(u)
        vn = np.copy(v)
        
        b = build_rhs(u, v, dx, dy, dt)
        p = pressure_poisson(p, b, dx, dy)
        update_velocity(u, v, p, dx, dy, dt, nu)
        
        if np.linalg.norm(u - un) < tolerance and np.linalg.norm(v - vn) < tolerance:
            print(f"Steady-state reached after {n} iterations")
            break
        
        if n % save_interval == 0:
            plot_velocity_magnitude_quiver(u, v, n)
    
    return u, v, p

def plot_velocity_profiles(u, v, Re):
    mid_x = int(N/2)
    mid_y = int(N/2)
    
    fig, ax = plt.subplots()
    ax.plot(u[:, mid_x], y)
    ax.set_xlabel('Vertical Velocity (u) at x=L/2')
    ax.set_ylabel('y')
    ax.set_title(f'Vertical Velocity Profile for Re = {Re}')
    ax.grid(True)

    add_annotations(ax)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, v[mid_y, :])
    ax.set_xlabel('x')
    ax.set_ylabel('Horizontal Velocity (v) at y=H/2')
    ax.set_title(f'Horizontal Velocity Profile for Re = {Re}')
    ax.grid(True)

    add_annotations(ax)
    plt.show()

def plot_streamlines(u, v, Re):
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, u, v, density=2, linewidth=1)
    ax.set_title(f'Streamlines for Re = {Re}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    add_annotations(ax)
    plt.show()

lid_driven_cavity_flow_with_plotting(Re_values[0])

for Re in Re_values:
    u, v, p = lid_driven_cavity_flow_with_plotting(Re)
    plot_velocity_profiles(u, v, Re)
    plot_streamlines(u, v, Re)
