# HW 2

This is the top level directory for HW assignment 2 of the Summer 2017 session of Parallel Algorithms at UT Austin.
The files in this direcotry include:

- common: routines called from several hw problems. Compiled into a library.
- p1: problem 1
- p2: problem 2
- p3: problem 3
- testCases: six fixed test inputs

## Building
This code has been built and tested on the UT TACC Stampede supercomputer environment. All code can be compiled simply by running:
```
module load cuda
make
```

## Running
Each problem can be run by submitting to the Stampede cluster. This can be done by moving to the proper problem directory and
either running `make submit`, or `sbatch submit`. The `submit` script for each problem is configured to run the problem 
for all of the test cases present in the `testCases` directory.
