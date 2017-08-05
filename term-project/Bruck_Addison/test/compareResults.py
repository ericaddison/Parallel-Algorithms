import numpy as np
import sys

# USAGE: <call> file1 file2
file1 = sys.argv[1]
file2 = sys.argv[2]

# load files
x1 = np.loadtxt(file1,dtype=float,skiprows=2,delimiter=',')
x2 = np.loadtxt(file2,dtype=float,skiprows=2,delimiter=',')

# compare and get stats
maxError = 0
rmsError = 0
x1rms = 0
print "n =", x1.size/2
for i in range(x1.size/2):
	x1irms = x1[i][0]**2 + x1[i][1]**2
	x1rms = x1rms + x1irms
	err = (x1[i][0]-x2[i][0])**2 + (x1[i][1]-x2[i][1])**2
	rmsError = rmsError + err
	maxError = max(err/x1irms,maxError)

rmsError = rmsError/x1rms

print "RMS Error =", rmsError
print "Max Error =", maxError
