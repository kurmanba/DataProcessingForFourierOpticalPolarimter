import numpy as np
from sympy import *
from matplotlib import pyplot as plt
import itertools

for i in itertools.combinations(range(100), 3):
    if i[0]**2 + i[1]**2 == i[2]**2:
        print(i)
