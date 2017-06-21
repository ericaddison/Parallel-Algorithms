CC = nvcc
NVCC_FLAGS = -arch=compute_35 -code=sm_35

ODIR = obj
SRC = hello.cu
_OBJ = $(SRC:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(OBJ)

$(ODIR)/%.o: $(SRC)
	$(CC) $(NVCC_FLAGS) $*.cu -c -o $@
	$(CC) $(NVCC_FLAGS) $@ -o $*.out
