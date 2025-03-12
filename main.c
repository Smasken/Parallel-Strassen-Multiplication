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
   if (size <= 64) { //Only uses strassen for 'large' matrices
      standard_matrix_multiplication(size, A, B, C);
      return;
   }
   int block_size = size / 2;

   // Allocate M1 to M7 and the submatrices for A and B

   data_type **M1=allocate_matrix(block_size);
   data_type **M2=allocate_matrix(block_size);
   data_type **M3=allocate_matrix(block_size);
   data_type **M4=allocate_matrix(block_size);
   data_type **M5=allocate_matrix(block_size);
   data_type **M6=allocate_matrix(block_size);
   data_type **M7=allocate_matrix(block_size);
   data_type **a11=allocate_matrix(block_size);
   data_type **a12=allocate_matrix(block_size);
   data_type **a21=allocate_matrix(block_size);
   data_type **a22=allocate_matrix(block_size);
   data_type **b11=allocate_matrix(block_size);
   data_type **b12=allocate_matrix(block_size);
   data_type **b21=allocate_matrix(block_size);
   data_type **b22=allocate_matrix(block_size);
   data_type **temp_result1 = allocate_matrix(block_size);
   data_type **temp_result2 = allocate_matrix(block_size);
   // Divides the input matrices into new ones

   for(int i = 0; i < block_size; i++) {
      for(int j = 0; j < block_size; j++) {
         a11[i][j] = A[i][j];
         a12[i][j] = A[i][j + block_size];
         a21[i][j] = A[i + block_size][j];
         a22[i][j] = A[i + block_size][j + block_size];
         b11[i][j] = B[i][j];
         b12[i][j] = B[i][j + block_size];
         b21[i][j] = B[i + block_size][j];
         b22[i][j] = B[i + block_size][j + block_size];
      }
   }

   /* ---- Calculate M1 to M7 ---- */ 
   // M1
   add_matrix(block_size, a11, a22, temp_result1);
   add_matrix(block_size, b11, b22, temp_result2);
   strassen_multiplication(block_size, temp_result1, temp_result2, M1);
   // M2
   add_matrix(block_size, a21, a22, temp_result1);
   strassen_multiplication(block_size, temp_result1, b11, M2);
   // M3
   subtract_matrix(block_size, b12, b22, temp_result1);
   strassen_multiplication(block_size, a11, temp_result1, M3);
   // M4
   subtract_matrix(block_size, b21, b11, temp_result1);
   strassen_multiplication(block_size, a22, temp_result1, M4);
   // M5
   add_matrix(block_size, a11, a12, temp_result1);
   strassen_multiplication(block_size, temp_result1, b22, M5);
   // M6
   subtract_matrix(block_size, a21, a11, temp_result1);
   add_matrix(block_size, b11, b12, temp_result2);
   strassen_multiplication(block_size, temp_result1, temp_result2, M6);
   // M7
   subtract_matrix(block_size, a12, a22, temp_result1);
   add_matrix(block_size, b21, b22, temp_result2);
   strassen_multiplication(block_size, temp_result1, temp_result2, M7);

   deallocate_matrix(block_size, a11);
   deallocate_matrix(block_size, a12);
   deallocate_matrix(block_size, a21);
   deallocate_matrix(block_size, a22);
   deallocate_matrix(block_size, b11);
   deallocate_matrix(block_size, b12);
   deallocate_matrix(block_size, b21);
   deallocate_matrix(block_size, b22);

   /* ---- Calculate C11 to C22 ---- */
   data_type ** c11=allocate_matrix(block_size);
   data_type ** c12=allocate_matrix(block_size);
   data_type ** c21=allocate_matrix(block_size);
   data_type ** c22=allocate_matrix(block_size);

   //c11
   add_matrix(block_size, M1, M4, temp_result1);
   add_matrix(block_size, temp_result1, M7, temp_result2);
   subtract_matrix(block_size, temp_result2, M5, c11);
   //c12
   add_matrix(block_size, M3, M5, c12);
   //c21
   add_matrix(block_size, M2, M4, c21);
   //c22
   add_matrix(block_size, M1, M3, temp_result1);
   add_matrix(block_size, temp_result1, M6, temp_result2);
   subtract_matrix(block_size, temp_result2, M2, c22);

   // Add together the submatrices

   for(int i = 0; i < block_size; i++) {
      for(int j = 0; j < block_size; j++) {
         C[i][j] = c11[i][j];
         C[i][j + block_size] = c12[i][j];
         C[i + block_size][j] = c21[i][j];
         C[i + block_size][j + block_size] = c22[i][j];
      }
   }

   // Deallocate submatrices
   deallocate_matrix(block_size, c11);
   deallocate_matrix(block_size, c12);
   deallocate_matrix(block_size, c21);
   deallocate_matrix(block_size, c22);
   deallocate_matrix(block_size, M1);
   deallocate_matrix(block_size, M2);
   deallocate_matrix(block_size, M3);
   deallocate_matrix(block_size, M4);
   deallocate_matrix(block_size, M5);
   deallocate_matrix(block_size, M6);
   deallocate_matrix(block_size, M7);
   deallocate_matrix(block_size, temp_result1);
   deallocate_matrix(block_size, temp_result2);
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

   strassen_multiplication(size, A, B, C);

   double end_time = get_wall_seconds();
   printf("Time taken: %f seconds\n", end_time - start_time);

   deallocate_matrix(size, A);
   deallocate_matrix(size, B);
   deallocate_matrix(size, C);

   return 0;
}