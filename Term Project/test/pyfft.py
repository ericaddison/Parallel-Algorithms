import numpy as np
import time
import sys

# step 0: write current algorithm to stdout
print "Python: numpy.fft.fft()"

# step 1: read input file
# file format:
#              # values
#              value 1
#              value 2
#              ...

y = np.fromfile(sys.argv[-1],dtype=float, sep='\n')[1:]


# step 2: perform fft with timing in microseconds
t0 = time.time()
y = np.fft.fft(y)
t1 = time.time()


# step 3: write elapsed time (microseconds) to stdout
print "elapsed time (microseconds):", (t1-t0)*1000000


# step 4: write result to stdout with format real, imag\n
for i in range(y.size):
    print y[i].real, ",", y[i].imag
