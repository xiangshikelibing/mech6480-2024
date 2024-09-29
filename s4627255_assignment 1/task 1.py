import numpy as np
import matplotlib.pyplot as plt
import time  
from datetime import datetime  
from git import Repo  


L = 0.1  
H = 0.01  
dpdx = 2.5 
mu = 1e-3  
q_flux = 5000  
rho = 1000  
cp = 4186  
k = 0.6  
ALPHA = k / (rho * cp)  

NX = 200  
NY = 20   
dx = L / NX 
dy = H / NY 

x = np.linspace(dx / 2, L - dx / 2, NX)  
y = np.linspace(dy / 2, H - dy / 2, NY)  
xx, yy = np.meshgrid(x, y, indexing='ij')

dt = 0.05  
total_time = 30  
nt = int(total_time / dt)


u_max = 0.0208  
u = u_max * (1 - ((y - H / 2) / (H / 2))**2)

T = np.ones((NX, NY)) * 300  
T_new = T.copy()


def compute_x_flux(T, u, dx, k, rho, cp):
    x_flux = np.zeros_like(T)
    for i in range(1, NX-1):
        for j in range(NY):
            convective_flux = u[j] * (T[i, j] - T[i-1, j]) / dx
            diffusive_flux = k * (T[i+1, j] - 2 * T[i, j] + T[i-1, j]) / dx**2
            x_flux[i, j] = (diffusive_flux - convective_flux) / (rho * cp)
    return x_flux

def compute_y_flux(T, dy, k, rho, cp):
    y_flux = np.zeros_like(T)
    for i in range(NX):
        for j in range(1, NY-1):
            diffusive_flux = k * (T[i, j+1] - 2 * T[i, j] + T[i, j-1]) / dy**2
            y_flux[i, j] = diffusive_flux / (rho * cp)
    return y_flux

def update_temperature(T, T_new, x_flux, y_flux, dt):
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (x_flux[1:-1, 1:-1] + y_flux[1:-1, 1:-1])
    
    for i in range(1, NX-1):
        T_new[i, 0] = T_new[i, 1] + q_flux * dy / k
        T_new[i, -1] = T_new[i, -2] - q_flux * dy / k
    
    T_new[-1, :] = T_new[-2, :]
    return T_new

time_points = [5, 10, 20, 30]
temperature_snapshots = {}

start_time = time.time()  

for n in range(nt):
    x_flux = compute_x_flux(T, u, dx, k, rho, cp)
    y_flux = compute_y_flux(T, dy, k, rho, cp)
    T_new = update_temperature(T, T_new, x_flux, y_flux, dt)
    T = T_new.copy()
    current_time = (n + 1) * dt
    if current_time in time_points:
        temperature_snapshots[current_time] = T.copy()

end_time = time.time()  
print(f"Total computation time: {end_time - start_time:.2f} seconds")

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

for time, temp_field in temperature_snapshots.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(temp_field.T, origin='lower', extent=[0, L, 0, H], aspect='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='Temperature (K)')
    ax.set_title(f'Temperature Field at t = {time} s')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    add_annotations(ax) 
    plt.show()

x_positions = [0.125 * L, 0.25 * L, 0.5 * L]
x_indices = [int(pos / dx) for pos in x_positions]

for x_idx, x_pos in zip(x_indices, x_positions):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, T[x_idx, :], label=f'x = {x_pos:.3f} m')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Vertical Temperature Profile at x = {x_pos:.3f} m at t = 30 s')
    ax.legend()

    add_annotations(ax)
    plt.show()

max_temp_over_time = [np.max(T) for T in temperature_snapshots.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_points, max_temp_over_time, '-o')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Maximum Temperature (K)')
ax.set_title('Maximum Temperature in the Domain Over Time')

add_annotations(ax)

plt.show()

print("Based on the plot of the maximum temperature over time, steady-state is", 
      "reached." if np.allclose(max_temp_over_time[-1], max_temp_over_time[-2], rtol=1e-3) else "not reached.")

