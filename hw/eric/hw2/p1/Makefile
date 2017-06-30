CC = nvcc
NVCC_FLAGS = -arch=compute_35 -code=sm_35
LIBS = -lm -lhw
INCLUDE_DIRS = -I../common/include
LIB_DIRS = -L../common

ODIR = obj
SDIR = src
_SRC = p1a.cu p1b.cu p1.cu
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ = $(_SRC:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(OBJ)
	$(CC) $(NVCC_FLAGS) $(LIB_DIRS) $(LIBS) $^ -o bin/p1.out

$(ODIR)/%.o: $(SDIR)/%.cu
	$(CC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cu -c -o $@

submit: all
	rm -f ea26799*
	sbatch submit
