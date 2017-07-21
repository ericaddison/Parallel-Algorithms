#!/bin/bash
echo "Starting Testing"

X=1

while [ $X -le 10000000 ]
do
  echo "Running with ${X} elements"
  ./rand_gen.py $X > input.txt
  sed -n 1p input.txt  | tr " " "\n"  | sort -n | tr "\n" " " > unix.txt
  Y=1
  while [ $Y -le 64 ]
  do
    echo "Executing with ${Y} Procs"
    mpiexec -n $Y ../bin/hw3_p2.out input.txt
    diff unix.txt sortedArray.txt
    rm sortedArray.txt
    Y=$[$Y*2]
  done
  echo ""
  X=$[$X*10]
done
echo "Finished testing"
