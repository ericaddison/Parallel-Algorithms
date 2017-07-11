CC = mpicxx
LIBS = -lm
INCLUDE_DIRS =
LIB_DIRS =

ODIR = obj
SDIR = src
#_SRC = matrix.cpp p1.cpp p1_funcs.cpp
_SRC = p2.cpp p2_funcs.cpp matrix.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ = $(_SRC:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(OBJ)
	$(CC) $(NVCC_FLAGS) $(LIB_DIRS) $(LIBS) $^ -o bin/hw3.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

run: all
	mpirun -n 4 ./bin/hw3.out ./data/testMat ./data/testVec

clean:
	rm -f obj/*
	rm -f bin/*
