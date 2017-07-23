CC = g++
CC_FLAGS = -std=c++11
LIBS = -lm
GTEST_LIBS = -lgtest -lpthread
INCLUDE_DIRS = -I../src
LIB_DIRS =

ODIR = obj
SDIR = src
_SRC = gtest_main.cpp complexTest.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ = $(_SRC:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

MAIN_ODIR = ../obj
_MAIN_OBJ = complex.o
MAIN_OBJ = $(patsubst %,$(MAIN_ODIR)/%,$(_MAIN_OBJ))

all: $(OBJ)
	$(CC) $(CC_FLAGS) $^ $(MAIN_OBJ) $(LIB_DIRS) $(LIBS) $(GTEST_LIBS) -o bin/test.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

clean:
	rm -f obj/*
	rm -f bin/*

#g++ -std=c++11  -lm -lgtest -lpthread obj/gtest_main.o obj/complexTest.o ../obj/complex.o -o bin/test.out

#g++ obj/gtest_main.o obj/complexTest.o ../obj/complex.o -lgtest -lpthread -std=c++11
