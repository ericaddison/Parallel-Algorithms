#!/usr/bin/env python
from __future__ import print_function
from random import randint
import sys

matrix = open("matrix.txt", 'w')
vector = open("vector.txt", 'w')

rows = int(sys.argv[1])
cols = int(sys.argv[2])

header = str(rows) + " " + str(cols) + "\n"

matrix.write(header)

r = int(rows)
c = int(cols)

string = ""
for row in xrange(r):
  for col in xrange(c):
    num = randint(-999,999)
    string = string + " " + str(num)
  string = string + '\n'

matrix.write(string)

string = ""
for col in xrange(c):
  num = randint(-999,999)
  string = string + " " + str(num)

vector.write(string)
