#!/bin/bash
echo "Starting mat-vec-mult Testing"

X=1

while [ $X -le 128 ]
do
  echo "Test #${X}"
  r=$((RANDOM%100+1))
  c=$((RANDOM%100+1))
  echo "r=${r} c=${c}"
  ./matrix_vector_gen.py $r $c
  Y=1
  while [ $Y -le 64 ]
  do
    echo "Executing with ${Y} Procs"
    mpiexec -n $Y ../bin/hw3_p1.out matrix.txt vector.txt
    Y=$[$Y*2+1]
  done
  X=$[$X*2]
done
echo "Finished testing"
