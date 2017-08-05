import numpy as np
import sys
import random

# get number of elements from command line
power = 4
if len(sys.argv)==2:
	arg = int(sys.argv[1])
	power = min(arg,30)
n = pow(2,power)

# write to stdout
print n

random.seed()
for i in range(n):
	nextNum = 200*random.random()-100
	print nextNum


