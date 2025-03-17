#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

// This allows me to change what to fill the matrix with (int, float, etc.)
typedef int data_type;

// Timing function
static double get_wall_seconds() {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec + (double) tv.tv_usec / 1000000;
}

// Allocates memory for a size*size matrix
data_type *allocate_matrix(int size) {
   data_type *matrix = (data_type *)calloc(size*size, sizeof(data_type));
   return matrix;
}

void fill_matrix(int size, data_type *matrix) {
   for (int i = 0; i < size*size; i++) {
           matrix[i] = (data_type)(rand() % 10);
   }
}

// Standard algorithm - multiplies A and B, C is output. This is borrowed from the lecture notes. 
void standard_matrix_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 2)
   for(int i = 0; i < size; i++) {
      for (int k = 0; k < size; k++) {
         for(int j = 0; j < size; j++) {
            C[i*size+j] += A[i*size+k] * B[k*size+j];
         }
      }
   }
}

void add_matrix(int size, data_type *A, data_type *B, data_type *C){
   #pragma omp parallel for
   for(int i = 0; i < size*size; i++) {
      C[i] = A[i] + B[i];
   }
}

void subtract_matrix(int size, data_type *A, data_type *B, data_type *C){
   #pragma omp parallel for
   for(int i = 0; i < size*size; i++) {
      C[i] = A[i] - B[i];
   }
}

void strassen_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   if (size <= 256) { //Only uses strassen when the (sub)matrices are large
      
      {
         standard_matrix_multiplication(size, A, B, C, num_threads);
      }
      
      return;
   }
   int block_size = size / 2;

   // Pointers to the submatrices for A and B, allocate memory for M1-M7

   data_type *a11 = A;
   data_type *a12 = A + block_size; 
   data_type *a21 = A + block_size * size; 
   data_type *a22 = A + block_size * (size + 1); 

   data_type *b11 = B;
   data_type *b12 = B + block_size;
   data_type *b21 = B + block_size * size;
   data_type *b22 = B + block_size * (size + 1);

   data_type *c11 = C;
   data_type *c12 = C + block_size;
   data_type *c21 = C + block_size * size;
   data_type *c22 = C + block_size * (size + 1);

   data_type *M1 = allocate_matrix(block_size);
   data_type *M2 = allocate_matrix(block_size);
   data_type *M3 = allocate_matrix(block_size);
   data_type *M4 = allocate_matrix(block_size);
   data_type *M5 = allocate_matrix(block_size);
   data_type *M6 = allocate_matrix(block_size);
   data_type *M7 = allocate_matrix(block_size);

   data_type *temp_result1 = allocate_matrix(block_size);
   data_type *temp_result2 = allocate_matrix(block_size);
   data_type *temp_result3 = allocate_matrix(block_size);
   data_type *temp_result4 = allocate_matrix(block_size);
   data_type *temp_result5 = allocate_matrix(block_size);
   data_type *temp_result6 = allocate_matrix(block_size);
   data_type *temp_result7 = allocate_matrix(block_size);
   data_type *temp_result8 = allocate_matrix(block_size);
   data_type *temp_result9 = allocate_matrix(block_size);
   data_type *temp_result10 = allocate_matrix(block_size);
   data_type *temp_result11 = allocate_matrix(block_size);
   data_type *temp_result12 = allocate_matrix(block_size);
   data_type *temp_result13 = allocate_matrix(block_size);
   data_type *temp_result14 = allocate_matrix(block_size);

   
   /* ---- Calculate M1 to M7 ---- */
   #pragma omp parallel num_threads(num_threads)
   {
      #pragma omp single nowait
      {
         // M1
         #pragma omp task
         {add_matrix(block_size, a11, a22, temp_result1);
         add_matrix(block_size, b11, b22, temp_result2);
         strassen_multiplication(block_size, temp_result1, temp_result2, M1, num_threads);}
         // M2
         #pragma omp task
         {add_matrix(block_size, a21, a22, temp_result3);
         strassen_multiplication(block_size, temp_result3, b11, M2, num_threads);}
         // M3
         #pragma omp task
         {subtract_matrix(block_size, b12, b22, temp_result4);
         strassen_multiplication(block_size, a11, temp_result4, M3, num_threads);}
         // M4
         #pragma omp task
         {subtract_matrix(block_size, b21, b11, temp_result5);
         strassen_multiplication(block_size, a22, temp_result5, M4, num_threads);}
         // M5
         #pragma omp task
         {add_matrix(block_size, a11, a12, temp_result6);
         strassen_multiplication(block_size, temp_result6, b22, M5, num_threads);}
         // M6
         #pragma omp task
         {subtract_matrix(block_size, a21, a11, temp_result7);
         add_matrix(block_size, b11, b12, temp_result8);
         strassen_multiplication(block_size, temp_result7, temp_result8, M6, num_threads);}
         // M7
         #pragma omp task
         {subtract_matrix(block_size, a12, a22, temp_result9);
         add_matrix(block_size, b21, b22, temp_result10);
         strassen_multiplication(block_size, temp_result9, temp_result10, M7, num_threads);}
         
         #pragma omp taskwait
      }
   }

   /* ---- Calculate C11 to C22 ---- */

   #pragma omp parallel num_threads(num_threads)
   {
      #pragma omp single nowait
      {
      //c11
         #pragma omp task
         {add_matrix(block_size, M1, M4, temp_result11);
         add_matrix(block_size, temp_result11, M7, temp_result12);
         subtract_matrix(block_size, temp_result12, M5, c11);}
         //c12
         #pragma omp task
         {add_matrix(block_size, M3, M5, c12);}
         //c21
         #pragma omp task
         {add_matrix(block_size, M2, M4, c21);}
         //c22
         #pragma omp task
         {add_matrix(block_size, M1, M3, temp_result13);
         add_matrix(block_size, temp_result13, M6, temp_result14);
         subtract_matrix(block_size, temp_result14, M2, c22);}

         #pragma omp taskwait
      }
   }

   free(M1);
   free(M2);
   free(M3);
   free(M4);
   free(M5);
   free(M6);
   free(M7);
   free(temp_result1);
   free(temp_result2);
   free(temp_result3);
   free(temp_result4);
   free(temp_result5);
   free(temp_result6);
   free(temp_result7);
   free(temp_result8);
   free(temp_result9);
   free(temp_result10);
}

int main(int argc, char *argv[]) {
   if (argc != 3) {
      printf("Input should be: ./a.out [matrix size] [num_threads]\n");
      return -1;
   }

   int size = atoi(argv[1]); //Input argument is matrix size
   int num_threads = atoi(argv[2]);

   data_type *A = allocate_matrix(size);
   data_type *B = allocate_matrix(size);
   data_type *C = allocate_matrix(size);

   fill_matrix(size, A);
   fill_matrix(size, B);

   double start_time = get_wall_seconds();

   strassen_multiplication(size, A, B, C, num_threads);

   double end_time = get_wall_seconds();
   printf("Time taken: %f seconds\n", end_time - start_time);

   // Used to check results. Compares a few entries in C with standard multiplication (by changing n_threshold)
      printf("%d %d %d\n", C[1], C[60], C[431]);

   free(A);
   free(B);
   free(C);

   return 0;
}