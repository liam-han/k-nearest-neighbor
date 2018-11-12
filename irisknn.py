import random
import math
import operator
import numpy as np

file = np.genfromtxt('breast-cancer-wisconsin.data', dtype=float, delimiter=',')
file = np.delete(file, 0, 1)
np.random.shuffle(file)
print(file)
n = file[~np.isnan(file).any(axis=1)]
p = np.array_split(n, 10)
#random.shuffle(p)
print(p[0])
distances = []
print (len(p))