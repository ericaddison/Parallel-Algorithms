#!/usr/bin/env python
from __future__ import print_function
from random import randint
import sys

X = int(sys.argv[1])
#s = str(X) + "\n"
#print(s, end='')

for x in range(0, X-1):
  i = randint(-999,999)
  s = str(i) + " "
  print(s, end='')
i = randint(2,999)
i = 1
print(i,  end='')
