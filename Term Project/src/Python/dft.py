import math
import numpy as np

def dft(x):
    N = x.size
    dft = np.zeros(N)
    w = math.exp(-2*1j*math.pi/N)
    for i in range(N):
        wi = math.pow(w,i)
        for j in range(N):
            dft[j] = dft[j] + x[j]*wi
    return dft

x = (np.random.rand(5)-0.5)*20
print x
X = dft(x)
print X
print np.fft.fft(x)
