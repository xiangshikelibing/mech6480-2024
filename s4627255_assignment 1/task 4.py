import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from git import Repo

L = 0.60
H = 0.03
step_height = 0.01
step_position = 0.22
rho = 998
mu = 0.001
Re = 230
U_in = Re * mu / (rho * step_height)

N_x = 300
N_y = 100
dx = L / (N_x - 1)
dy = H / (N_y - 1)
dt = 0.0001
max_iter = 10000
tolerance = 1e-6
save_interval = 500

x = np.linspace(0, L, N_x)
y = np.linspace(0, H, N_y)
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
    u = np.zeros((N_x, N_y))
    v = np.zeros((N_x, N_y))
    p = np.zeros((N_x, N_y))
    u[0, int(step_height/dy):] = U_in
    return u, v, p

def apply_boundary_conditions(u, v):
    u[0, int(step_height/dy):] = U_in
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = u[-2, :]
    
    v[0, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    v[-1, :] = v[-2, :]
    
    u[int(step_position/dx):, int(step_height/dy)] = 0
    v[int(step_position/dx):, int(step_height/dy)] = 0

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
    b = np.zeros((N_x, N_y))
    b[1:-1, 1:-1] = (rho * (1/dt * 
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) + 
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx))**2 -
                     2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy) * 
                          (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy))**2))
    
    b = np.clip(b, -1e10, 1e10)
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
    cont = ax.contourf(X, Y, velocity_magnitude.T, cmap='viridis')
    plt.colorbar(cont, ax=ax, label='Velocity Magnitude')
    ax.quiver(X[::5, ::5], Y[::5, ::5], u.T[::5, ::5], v.T[::5, ::5], scale=5)
    ax.set_title(f'Velocity Field at iteration {n}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    add_annotations(ax)
    
    plt.savefig(f'velocity_field_{n}.png')
    plt.close()

def backward_facing_step_flow():
    nu = U_in * step_height / Re
    
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

def plot_velocity_profiles(u):
    x_22_index = int(0.22 / dx)
    x_28_index = int(0.28 / dx)
    
    fig, ax = plt.subplots()
    ax.plot(u[x_22_index, :], np.linspace(0, H, N_y), label='x=0.22m')
    ax.plot(u[x_28_index, :], np.linspace(0, H, N_y), label='x=0.28m')
    ax.set_xlabel('Horizontal velocity (m/s)')
    ax.set_ylabel('Height (m)')
    ax.legend()
    
    add_annotations(ax)
    
    plt.show()

def plot_streamlines(u, v):
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, u.T, v.T, density=2, linewidth=1)
    ax.set_title('Streamlines')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    add_annotations(ax)
    
    plt.show()

u, v, p = backward_facing_step_flow()

plot_velocity_profiles(u)
plot_streamlines(u, v)

