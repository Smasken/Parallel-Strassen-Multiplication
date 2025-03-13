# Compiler and flags
CC = gcc
CFLAGS = -O3 -g -march=native -ffast-math -ftree-vectorize -funroll-loops #-fopt-info-vec-optimized 
LDFLAGS = 

# Target
EXECUTABLE = strassen
SOURCE = main.c

# Build rules
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE) $(LDFLAGS)

clean:
	/bin/rm -f $(EXECUTABLE)