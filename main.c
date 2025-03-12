#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

// Timing function
static double get_wall_seconds() {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec + (double) tv.tv_usec / 1000000;
}
// This allows me to change what to fill the matrix with (int, float, etc.)
typedef float data_type;

// Allocates memory for a size*size matrix
data_type **allocate_matrix(int size) {
   data_type **matrix = (data_type**)malloc(sizeof(data_type*) * size);
   for(int i = 0; i < size; i++) {
      matrix[i] = (data_type*)malloc(sizeof(data_type) * size);
   }
   return matrix;
}

void fill_matrix(int size, data_type **matrix) {
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            matrix[i][j] = (data_type)(rand() % 1000);
        }
    }
}

void deallocate_matrix(int size, data_type **matrix) {
   for(int i = 0; i < size; i++) {
      free(matrix[i]);
   }
   free(matrix);
}

// Standard algorithm - multiplies A and B, C is output
void standard_matrix_multiplication(int size, data_type **A, data_type **B, data_type **C) {
   for(int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         C[i][j]=0;
         for(int k = 0; k < size; k++) {
            C[i][j] += A[i][k] * B[k][j];
         }
      }
   }
}

void add_matrix(int size, data_type **A, data_type **B, data_type **C){
   for(int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         C[i][j] = A[i][j] + B[i][j];
      }
   }
}

void subtract_matrix(int size, data_type **A, data_type **B, data_type **C){
   for(int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         C[i][j] = A[i][j] - B[i][j];
      }
   }
}

void strassen_multiplication(int size, data_type **A, data_type **B, data_type **C) {

}

int main(int argc, char *argv[]) {
   if (argc != 2) {
      printf("Input should be: ./a.out [matrix size]\n");
      return -1;
   }

   int size = atoi(argv[1]); //Input argument is matrix size

   data_type **A = allocate_matrix(size);
   data_type **B = allocate_matrix(size);
   data_type **C = allocate_matrix(size);

   fill_matrix(size, A);
   fill_matrix(size, B);

   double start_time = get_wall_seconds();

   // Strassen only used for large matrices
   if(size < 256) {
      standard_matrix_multiplication(size, A, B, C);
   } else {
      strassen_multiplication(size, A, B, C);
   }

   double end_time = get_wall_seconds();
   printf("Time taken: %f seconds\n", end_time - start_time);

   deallocate_matrix(size, A);
   deallocate_matrix(size, B);
   deallocate_matrix(size, C);

   return 0;
}