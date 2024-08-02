# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:14:11 2024

@author: 13747
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('curve.data', 'r') as i:
    rawdata = list(csv.reader(i,delimiter = ","))
d =rawdata[1:]
x = []
y = []
for i in d:
    x.append(float(i[0].split("  ")[0]))
    y.append(float(i[0].split("  ")[1]))

plt.plot(x, y)
plt.show()
