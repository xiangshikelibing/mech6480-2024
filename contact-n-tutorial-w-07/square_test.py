# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:51:22 2024

@author: 13747
"""

from path import Line
from vector3 import Vector3
from gridutils import plot_grid
import numpy as np

p00 = Vector3(0.0,0.0)
p10 = Vector3(2.0,0.0)
p11 = Vector3(2.0,2.0)
p01 = Vector3(0.0,2.0)

north = Line((p01),(p11))
east = Line((p10),(p11))
south = Line((p00),(p10))
west = Line((p00),(p01))

def coons_patch_grid(north, east, south, west, niv, njv):
    points = [None]*niv
    for i in range(niv): points[i] = [None]*njv
    rs = np.linspace(0, 1, niv, endpoint=True)
    ss = np.linspace(0, 1, njv, endpoint=True)
    p00 = south(0,0)
    p10 = south(1,0)
    p01 = north(0,0)
    p11 = north(1,0)
    for i, r in enumerate(rs):
        for j, s in enumerate(ss):
            #first term (iw) F_i(u)
            first_term = west(s)*(1. - r) +east(s)*r
            second_term = south(r)(1. - s) +north(r)*s
            third_term = p00*(1. - r)(1. - s) + p10*r*(1. - s) + p01*(1. - r)*s + p11*r*s
            points[i][j] = first_term + second_term - third_term
    return points

my_grid_points = coons_patch_grid(north, east, south, west, 5, 8)

plot_grid(my_grid_points,x_lim=(0, 2), y_lim=(0, 2), filename='square.png')
    