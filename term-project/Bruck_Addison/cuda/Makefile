CC = g++
NVCC = nvcc
CC_FLAGS = -std=c++14
NVCC_FLAGS = -arch=compute_35 -code=sm_35
LIBS = -lm -lcufft
INCLUDE_DIRS = -I../cpp/src -I/opt/apps/cuda/7.0/include
LIB_DIRS = -L/opt/apps/cuda/7.0/lib64

ODIR = obj
SDIR = src
_COMMON_SRC = ft_helpers.cpp 
_MYCUDA_SRC = main.cpp fft_cuda.cpp fft_cuda_kernels.cu
_CUFFT_SRC = main_cufft.cpp fft_cufft.cu
_COMMON_OBJ1 = $(_COMMON_SRC:.cpp=.o)
_COMMON_OBJ2 = $(_COMMON_OBJ1:.cu=.o)
_MYCUDA_OBJ1 = $(_MYCUDA_SRC:.cpp=.o)
_MYCUDA_OBJ2 = $(_MYCUDA_OBJ1:.cu=.o)
_CUFFT_OBJ1 = $(_CUFFT_SRC:.cpp=.o)
_CUFFT_OBJ2 = $(_CUFFT_OBJ1:.cu=.o)


COMMON_SRC = $(patsubst %,$(SDIR)/%,$(_COMMON_SRC))
MYCUDA_SRC = $(patsubst %,$(SDIR)/%,$(_MYCUDA_SRC))
CUFFT_SRC = $(patsubst %,$(SDIR)/%,$(_CUFFT_SRC))

COMMON_OBJ = $(patsubst %,$(ODIR)/%,$(_COMMON_OBJ2))
MYCUDA_OBJ = $(patsubst %,$(ODIR)/%,$(_MYCUDA_OBJ2))
CUFFT_OBJ = $(patsubst %,$(ODIR)/%,$(_CUFFT_OBJ2))

all: mycuda cufft

mycuda: $(COMMON_OBJ) $(MYCUDA_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_mycuda.out

cufft: $(COMMON_OBJ) $(CUFFT_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_cufft.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

$(ODIR)/%.o: $(SDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cu -c -o $@

submit: all
	rm -f result*
	sbatch submit

clean:
	rm -f obj/*
	rm -f bin/*
