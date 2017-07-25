CC = g++
CC_FLAGS = -std=c++14
LIBS = -lm
INCLUDE_DIRS =
LIB_DIRS =

ODIR = obj
SDIR = src
_SRC = main.cpp dft.cpp fft.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ = $(_SRC:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(OBJ)
	$(CC) $(CC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/a.out

test: all
	make -C test

runTest: test
	./test/bin/test.out


$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@


clean:
	make -C test clean
	rm -f obj/*
	rm -f bin/*
