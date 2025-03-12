#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>

// This allows me to change what to fill the matrix with (int, float, etc.)
typedef float data_type;

// This function allocates memory for a size*size matrix and fills it with random floats.
data_type **allocate_matrix(int size) {
   data_type **matrix = (data_type**)malloc(sizeof(data_type)*size);
   for(int i = 0; i < size; i++) {
      matrix[i] = (data_type*)malloc(sizeof(data_type)*size);
   }

   for(int i = 0; i < size; i++) { // Fills matrix
      for(int j = 0; j < size; j++) {
          matrix[i][j] = rand() % 100;
      }
  }
  return matrix;
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
      for (int j = 0; j < size; i++) {
         C[i][j] = A[i][j] + B[i][j];
      }
   }
}

void subtract_matrix(int size, data_type **A, data_type **B, data_type **C){
   for(int i = 0; i < size; i++) {
      for (int j = 0; j < size; i++) {
         C[i][j] = A[i][j] - B[i][j];
      }
   }
}

void strassen_multiplication(int size, data_type **A, data_type **B, data_type **C) {
   
}

int main(){

}