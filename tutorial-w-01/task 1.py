# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:01:25 2024

@author: 13747
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
L = 0.1  # Length of the channel (m)
H = 0.01  # Height of the channel (m)
dpdx = 0.0025  # Pressure gradient (Pa/m)
mu = 1e-3  # Dynamic viscosity (Pa.s)
q_flux = 5000  # Heat flux (W/m^2)
rho = 1000  # Density (kg/m^3)
cp = 4186  # Specific heat capacity (J/kg.K)
k = 0.6  # Thermal conductivity (W/m.K)
ALPHA = k / (rho * cp)  # Thermal diffusivity (m^2/s)

# Grid parameters
NX = 200  # Number of grid points in x
NY = 20   # Number of grid points in y
dx = L / NX  # Grid spacing in x
dy = H / NY  # Grid spacing in y

# Control volume centers
x = np.linspace(dx / 2, L - dx / 2, NX)  # C.V. centers in x
y = np.linspace(dy / 2, H - dy / 2, NY)  # C.V. centers in y
xx, yy = np.meshgrid(x, y, indexing='ij')

print(f"dx={dx:.6f}, dy={dy:.6f}")

# Time-stepping parameters
dt = 0.05  # Time step (s)
total_time = 30  # Total simulation time (s)
nt = int(total_time / dt)

# Velocity profile (Poiseuille flow) at C.V. centers in y
u = (1 / (2 * mu) * dpdx * ((H / 2)**2 - (y - H / 2)**2))

# Initialize temperature field
T = np.ones((NX, NY)) * 300  # Initial temperature in the domain (300 K)
T_new = T.copy()

# Function to update temperature field
def update_temperature(T, T_new, u, dt, dx, dy, rho, cp, k, q_flux):
    for i in range(1, NX-1):
        for j in range(1, NY-1):
            # Diffusive term
            diffusion_x = k * (T[i+1, j] - 2 * T[i, j] + T[i-1, j]) / dx**2
            diffusion_y = k * (T[i, j+1] - 2 * T[i, j] + T[i, j-1]) / dy**2

            # Convective term
            convection = u[j] * (T[i, j] - T[i-1, j]) / dx

            # Update temperature
            T_new[i, j] = T[i, j] + dt * ((diffusion_x + diffusion_y) / (rho * cp) - convection)

        # Neumann boundary conditions (heat flux at top and bottom)
        T_new[i, 0] = T_new[i, 1] + q_flux * dy / k
        T_new[i, -1] = T_new[i, -2] - q_flux * dy / k

    # Neumann condition at outlet (zero gradient)
    T_new[-1, :] = T_new[-2, :]

    return T_new

# Time-stepping loop
time_points = [5, 10, 20, 30]
temperature_snapshots = {}

for n in range(nt):
    T_new = update_temperature(T, T_new, u, dt, dx, dy, rho, cp, k, q_flux)
    T = T_new.copy()

    # Save snapshots at specific times
    current_time = (n + 1) * dt
    if current_time in time_points:
        temperature_snapshots[current_time] = T.copy()

# Plot heat maps at specified times
for time, temp_field in temperature_snapshots.items():
    plt.figure(figsize=(8, 5))
    plt.imshow(temp_field.T, origin='lower', extent=[0, L, 0, H], aspect='auto', cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.title(f'Temperature Field at t = {time} s')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

# Vertical profiles of temperature at x = {0.125L, 0.25L, 0.5L}
x_positions = [0.125 * L, 0.25 * L, 0.5 * L]
x_indices = [int(pos / dx) for pos in x_positions]

# Separate plots for each vertical profile
for x_idx, x_pos in zip(x_indices, x_positions):
    plt.figure(figsize=(8, 5))
    plt.plot(y, T[x_idx, :], label=f'x = {x_pos:.3f} m')
    plt.xlabel('y (m)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Vertical Temperature Profile at x = {x_pos:.3f} m at t = 30 s')
    plt.legend()
    plt.show()

# Plot maximum temperature in the domain over time
max_temp_over_time = [np.max(T) for T in temperature_snapshots.values()]
plt.figure(figsize=(8, 5))
plt.plot(time_points, max_temp_over_time, '-o')
plt.xlabel('Time (s)')
plt.ylabel('Maximum Temperature (K)')
plt.title('Maximum Temperature in the Domain Over Time')
plt.show()

# Comment on steady-state
print("Based on the plot of the maximum temperature over time, steady-state is", 
      "reached." if np.allclose(max_temp_over_time[-1], max_temp_over_time[-2], rtol=1e-3) else "not reached.")
