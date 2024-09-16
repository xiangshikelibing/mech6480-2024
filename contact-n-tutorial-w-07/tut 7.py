# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:58:18 2024

@author: 13747
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from git import Repo
from path import Line, Bezier, Arc
from vector3 import Vector3

# Step 1: Define the points and boundaries
p00 = Vector3(0.0, 0.0)  
p10 = Vector3(1.0, 0.0)  
p20 = Vector3(2.0, 0.0)  
p21 = Vector3(2.0, 1.0)  
p01 = Vector3(0.0, 2.0)  

arc_center = Vector3(2.0, 0.0)  

south = Line(p00, p10)
east = Arc(p10, p21, arc_center)
north = Bezier([p01, Vector3(0.7, 2.5), p21])
west = Bezier([p00, Vector3(0.7, 1.0), p01])

# Step 2: Define the Coons patch grid function
def coons_patch_grid(north, east, south, west, niv, njv):
    points = [None] * niv
    for i in range(niv):
        points[i] = [None] * njv
    rs = np.linspace(0, 1, niv, endpoint=True)
    ss = np.linspace(0, 1, njv, endpoint=True)
    p00 = south(0.0)
    p10 = south(1.0)
    p01 = north(0.0)
    p11 = north(1.0)
    for i, r in enumerate(rs):
        for j, s in enumerate(ss):
            first_term = west(s) * (1.0 - r) + east(s) * r
            second_term = south(r) * (1.0 - s) + north(r) * s
            third_term = p00 * (1.0 - r) * (1.0 - s) + p10 * r * (1.0 - s) + p01 * (1.0 - r) * s + p11 * r * s
            points[i][j] = first_term + second_term - third_term
    return points

# Step 3: Define the Laplace smoothing filter function
def smooth_grid(vertices, nsweeps):
    niv = len(vertices)      
    njv = len(vertices[0])   
    vertices_np = np.array(vertices)
    smoothed_vertices = np.copy(vertices_np)
    for sweep in range(nsweeps):
        for i in range(1, niv - 1):
            for j in range(1, njv - 1):
                smoothed_vertices[i, j] = 0.25 * (
                    vertices_np[i + 1, j] +
                    vertices_np[i - 1, j] +
                    vertices_np[i, j + 1] +
                    vertices_np[i, j - 1]
                )
        vertices_np = np.copy(smoothed_vertices)
    for i in range(niv):
        for j in range(njv):
            vertices[i][j] = smoothed_vertices[i][j]

# Step 4: Generate the grid
my_grid_points = coons_patch_grid(north, east, south, west, 21, 21)

# Step 5: Apply the Laplace smoothing filter with 500 sweeps
smooth_grid(my_grid_points, 500)


fig, ax = plt.subplots(figsize=(6, 6))  

for i in range(len(my_grid_points)):
    for j in range(len(my_grid_points[i]) - 1):
        ax.plot([my_grid_points[i][j].x, my_grid_points[i][j + 1].x],
                [my_grid_points[i][j].y, my_grid_points[i][j + 1].y], 'b-', lw=0.5)

for j in range(len(my_grid_points[0])):
    for i in range(len(my_grid_points) - 1):
        ax.plot([my_grid_points[i][j].x, my_grid_points[i + 1][j].x],
                [my_grid_points[i][j].y, my_grid_points[i + 1][j].y], 'b-', lw=0.5)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2.5)
ax.set_aspect('equal')


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ax.annotate(timestamp, xy=(0.7, 0.95), xycoords='axes fraction', annotation_clip=False)

try:
    repo = Repo('.', search_parent_directories=True)
    revsha = repo.head.object.hexsha[:8]
    ax.annotate(f"[rev {revsha}]", xy=(0.05, 0.95), xycoords='axes fraction', annotation_clip=False)
except Exception as e:
    ax.annotate("[rev unknown]", xy=(0.05, 0.95), xycoords='axes fraction', annotation_clip=False)
    print(f"Error accessing Git repository: {e}")


plt.show()
