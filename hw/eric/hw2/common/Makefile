CC = gcc
CFLAGS = -std=c99 -fPIC
LIBS = -lm

ODIR = obj
SDIR = src
_SRC = readIntFile.c usefulFuncs.c
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ = $(_SRC:.c=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_DEPS = randomInts.h
DEPS = $(patsubst %,$(SDIR)/%,$(_DEPS)

all: $(OBJ)
	ar rcs libhw.a $^

$(ODIR)/%.o: $(SRC)
	$(CC) $(CFLAGS) $(LIBS) $(SDIR)/$*.c -c -o $@
