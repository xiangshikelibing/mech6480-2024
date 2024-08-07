# -*- coding: utf-8 -
"""
Created on Wed Aug  7 10:27:20 2024

@author: 13747
"""
import numpy as np
import matplotlib.pyplot as plt

k = 10
rho_c = 10 * 10 ** 6
length = 0.02
total_time = 120

T_ic = 200

dT_A = 0

T_B = 0

num_cells = 5
dx = length / num_cells
x_locations = np.linspace(0.5 * dx, (num_cells - 0.5)* dx, num_cells)
dt = 5
steps = int(total_time/ dt) + 1

temperature = T_ic * np.ones(num_cells)
temperature_buffer = temperature.copy()

plt.plot(x_locations, temperature, 'r-o', label ='0s')
plt.plot([0, length], [temperature[0], T_B], 'ko', label = 'boundary')

for step in range(steps):
    for cell in range(1, num_cells -1):
        temperature_buffer[cell] = temperature[cell] + (k * dt/ (rho_c * dx ** 2)) *\
            (temperature[cell+1] - 2 * temperature[cell] + temperature[cell-1])
            
    temperature_buffer[0] = temperature[0] + (k * dt/ (rho_c* dx ** 2)) *\
        (temperature[1] - temperature[0])
        
    temperature_buffer[-1] = temperature[-1] + (k * dt/ (rho_c* dx ** 2)) *\
        (2 * T_B - 3 * temperature[-1] + temperature[-2])
        
    temperature = temperature_buffer.copy()
    
    if ((dt*step) >= 60 and (dt * (step-1)< 60)) or ((dt*step) >= 80 and (dt * (step-1)< 80)) or ((dt*step) >= 120 and (dt * (step-1)< 120)):
        print(dt*step)
        plt.plot([0,length], [temperature, 'b-o'], label = str(dt * steps) +'s')
        plt.plot([0, length],[temperature[0], T_B], 'ko')

plt.legend()
        
    
