
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from git import Repo

def generate_matrix_dirichlet(n,dx,convective_flux, diffusive_conductance, boundary_left, boundary_right):
    """ Generate solution matrix for the FVM """
    
    a_WW = -1/8 * F
    a_W = D + 6/8 * F
    a_E = D - 3/8 * F
    a_P = a_W + a_E + a_WW

    # In this case, our coefficients are constant for all internal cells
    # This means we can use np.repeat to generate the diagonals
    matrix_A = np.diag(np.repeat(-a_WW,n-2),-2) +\
               np.diag(np.repeat(-a_W,n-1),-1) +\
               np.diag(np.repeat(a_P,n),0) +\
               np.diag(np.repeat(-a_E,n-1),1)
    
    S_p_left  = - (8/3 * D + F)
    S_p_right = - (8/3 * D - F)

    # Left boundary CV
    matrix_A[0,0] = a_E + S_p_left
    matrix_A[0,1] =- a_E

    # One CV in from left boundary CV
    matrix_A[1,0] = - a_W 
    matrix_A[1,2] = - a_E
    matrix_A[1,1] = a_P
    
    # Right boundary CV
    matrix_A[-1,-2] = - a_W 
    matrix_A[-1,-1] = a_W + S_p_right

    vector_b = np.zeros(n)
    vector_b[0]  = S_p_left * boundary_left
    vector_b[1]  = 0
    vector_b[-1] = S_p_right * boundary_right
    print(matrix_A)
    print(vector_b.T)
    return matrix_A, vector_b.T

# SYSTEM PARAMETERS
Gamma = 0.1             # kg/m.s      
u = 0.2                 # m/s
rho = 1.0               # kg/m3
length = 1              # m

# BOUNDARIES:
phi_0 = 1
phi_L = 0

# GRID GENERATION
num_cells = 16              #[-]
dx = length / (num_cells)  #[m]
x_locations = np.linspace(0.5*dx,(num_cells-0.5)*dx,num_cells)

# CONVECTIVE AND DIFFUSIVE TERMS
#   note: you may need to calculate these locally if rho, u, Gamma, or dx is not consistent in the domain.
F = rho * u
D = Gamma / dx

# SYSTEM OF EQUATIONS
solution_matrix, source_matrix = generate_matrix_dirichlet(num_cells, dx, F, D, phi_0, phi_L)
concentration = np.linalg.solve(solution_matrix, source_matrix)

print(solution_matrix)
print ("")
print(source_matrix)
def analytic_solution(x):
    return phi_0 + (phi_L - phi_0) * (np.exp(rho*u*x/Gamma)-1) / (np.exp(rho*u*length/Gamma) -1)
x = np.linspace(0,length,100)
solution = analytic_solution(x)
plt.plot(x, solution, 'r--')
        
plt.plot(x_locations, concentration, 'b-o')
plt.xlabel('Distance along hallway (m)')
plt.ylabel('concentration')
plt.show()

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='figure fraction', annotation_clip=False)


try:
    repo = Repo('.', search_parent_directories=True)
    revsha = repo.head.object.hexsha[:8]
    ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
except Exception as e:
    ax.annotate("[rev unknown]", xy=(0.05, 0.95), xycoords='figure fraction', annotation_clip=False)
    print(f"Error accessing Git repository: {e}")
    
plt.show()