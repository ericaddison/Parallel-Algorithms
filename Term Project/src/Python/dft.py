import math
import numpy as np

def dft(x):
    N = x.size;
    dft = np.zeros(N)
    w = math.exp(-2*math.pi/N)
    for i in range(N):
        print i

x = (np.random.rand(5)-0.5)*20
print x
dft(x)
