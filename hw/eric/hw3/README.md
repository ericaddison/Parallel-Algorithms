# HW 3
Eric Addison and Ari Bruck

This is the top level directory for HW assignment 3 of the Summer 2017 session of Parallel Algorithms at UT Austin.
The files in this direcotry include:

- bin: executable outputs
- data: test data input files
- Makefile: the makefile for both problems
- obj: compiled object files
- README.md: this file
- src: C++ source files

## Building
This code has been built and tested on Linux Mint 18 with Mpich. All code can be compiled simply by running:
```bash
make	   # build executables for both problems
make run1  # build all and run problem 1 with sample data
make run2  # build all and run problem 2 with sample data
```

## Running
Each problem can be run with the `mpirun` or `mpiexec` programs. The problems can be run as in the following examples:
```bash
mpirun -n 8 ./bin/hw3_p1.out ./data/matrix-1.txt ./data/vector-1.txt

```

## Input Files
Samples for the matrix and vector inputs were provided on github. Matrix files contain a one line header with explicit dimensions. Vector files contain a space separated list of values. For the sorting problem, we used the same format as the vectors from problem 1: a single line, space separated list of values.
