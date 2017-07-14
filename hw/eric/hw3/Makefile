CC = mpicxx
LIBS = -lm
INCLUDE_DIRS =
LIB_DIRS =

ODIR = obj
SDIR = src
#_SRC = matrix.cpp
_SRC_p1 = p1.cpp p1_funcs.cpp matrix.cpp
SRC_p1 = $(patsubst %,$(SDIR)/%,$(_SRC_p1))
_OBJ_p1 = $(_SRC_p1:.cpp=.o)
OBJ_p1 = $(patsubst %,$(ODIR)/%,$(_OBJ_p1))

_SRC_p2 = p2.cpp quickSort.cpp matrix.cpp ParallelHyperQuickSorter.cpp
SRC_p2 = $(patsubst %,$(SDIR)/%,$(_SRC_p2))
_OBJ_p2 = $(_SRC_p2:.cpp=.o)
OBJ_p2 = $(patsubst %,$(ODIR)/%,$(_OBJ_p2))


all: p1 p2

p1: $(OBJ_p1)
	$(CC) $(NVCC_FLAGS) $(LIB_DIRS) $(LIBS) $^ -o bin/hw3_$@.out

p2: $(OBJ_p2)
	$(CC) $(NVCC_FLAGS) $(LIB_DIRS) $(LIBS) $^ -o bin/hw3_$@.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

run1: p1
	mpirun -n 8 ./bin/hw3_p1.out ./data/matrix-1.txt ./data/vector-1.txt

run2: p2
	mpirun -n 8 ./bin/hw3_p2.out ./data/vector-1.txt

clean:
	rm -f obj/*
	rm -f bin/*
