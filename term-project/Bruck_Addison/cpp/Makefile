CC = g++
CC_FLAGS = -std=c++14
LIBS = -lm
INCLUDE_DIRS = -I./src
LIB_DIRS =

ODIR = obj
SDIR = src

_DFT_SRC = main_dft.cpp dft.cpp
_FFT_REC_SRC = main_fft_recursive.cpp fft_recursive.cpp
_FFT_ITER_SRC = main_fft_iterative.cpp fft_iterative.cpp
_COMMON_SRC = ft_helpers.cpp
_DFT_OBJ = $(_DFT_SRC:.cpp=.o)
_FFT_REC_OBJ = $(_FFT_REC_SRC:.cpp=.o)
_FFT_ITER_OBJ = $(_FFT_ITER_SRC:.cpp=.o)
_COMMON_OBJ = $(_COMMON_SRC:.cpp=.o)

DFT_SRC = $(patsubst %,$(SDIR)/%,$(_DFT_SRC))
FFT_REC_SRC = $(patsubst %,$(SDIR)/%,$(_FFT_REC_SRC))
FFT_ITER_SRC = $(patsubst %,$(SDIR)/%,$(_FFT_ITER_SRC))
COMMON_SRC = $(patsubst %,$(SDIR)/%,$(_COMMON_SRC))

DFT_OBJ = $(patsubst %,$(ODIR)/%,$(_DFT_OBJ))
FFT_REC_OBJ = $(patsubst %,$(ODIR)/%,$(_FFT_REC_OBJ))
FFT_ITER_OBJ = $(patsubst %,$(ODIR)/%,$(_FFT_ITER_OBJ))
COMMON_OBJ = $(patsubst %,$(ODIR)/%,$(_COMMON_OBJ))


all:  dft fft_rec fft_iter

dft: $(COMMON_OBJ) $(DFT_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/dft_seq.out
	
fft_rec: $(COMMON_OBJ) $(FFT_REC_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_rec_seq.out

fft_iter: $(COMMON_OBJ) $(FFT_ITER_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_iter_seq.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

clean:
	rm -f obj/*
	rm -f bin/*
