CC = icc
CC_FLAGS = -std=c++14 -ldl
LIBS = -lm
INCLUDE_DIRS =
LIB_DIRS =

ODIR = obj
SDIR = src

_CILK_FFT_REC_SRC = main_cilk_recursive.cpp fft.cpp
_CILK_FFT_ITER_SRC = main_cilk_iterative.cpp fft.cpp
_CILK_FFT_MINE_SRC = main.cpp fft.cpp
_COMMON_SRC = ft_helpers.cpp

_CILK_FFT_REC_OBJ = $(_CILK_FFT_REC_SRC:.cpp=.o)
_CILK_FFT_ITER_OBJ = $(_CILK_FFT_ITER_SRC:.cpp=.o)
_CILK_FFT_MINE_OBJ = $(_CILK_FFT_MINE_SRC:.cpp=.o)
_COMMON_OBJ = $(_COMMON_SRC:.cpp=.o)

CILK_FFT_REC_SRC = $(patsubst %,$(SDIR)/%,$(_CILK_FFT_REC_SRC))
CILK_FFT_ITER_SRC = $(patsubst %,$(SDIR)/%,$(_CILK_FFT_ITER_SRC))
CILK_FFT_MINE_SRC = $(patsubst %,$(SDIR)/%,$(_CILK_FFT_MINE_SRC))
COMMON_SRC = $(patsubst %,$(SDIR)/%,$(_COMMON_SRC))

CILK_FFT_REC_OBJ = $(patsubst %,$(ODIR)/%,$(_CILK_FFT_REC_OBJ))
CILK_FFT_ITER_OBJ = $(patsubst %,$(ODIR)/%,$(_CILK_FFT_ITER_OBJ))
CILK_FFT_MINE_OBJ = $(patsubst %,$(ODIR)/%,$(_CILK_FFT_MINE_OBJ))
COMMON_OBJ = $(patsubst %,$(ODIR)/%,$(_COMMON_OBJ))


all:  cilk_fft_rec cilk_fft_iter

cilk_fft_rec: $(COMMON_OBJ) $(CILK_FFT_REC_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_rec_cilk.out

cilk_fft_iter: $(COMMON_OBJ) $(CILK_FFT_ITER_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_iter_cilk.out

cilk_fft_mine: $(COMMON_OBJ) $(CILK_FFT_MINE_OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_mine_cilk.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

clean:
	rm -f obj/*
	rm -f bin/*
