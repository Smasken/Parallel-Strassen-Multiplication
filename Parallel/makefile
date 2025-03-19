# Compiler and flags
CC = gcc
CFLAGS = -O3 -g -fopenmp -march=native -ffast-math -ftree-vectorize -funroll-loops #-fopt-info-vec-optimized 
LDFLAGS = -fopenmp

# Target
EXECUTABLE = strassen
SOURCE = strassen_final.c

# Build rules
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE) $(LDFLAGS)

clean:
	/bin/rm -f $(EXECUTABLE)
