# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:14:11 2024

@author: 13747
"""
import csv
import numpy as np

with open('curve.data', 'r') as i:
    rawdata = list(csv.reader(i,delimiter = ","))

exampledata = np.array(rawdata[1:])

xdata = exampledata[:,0]