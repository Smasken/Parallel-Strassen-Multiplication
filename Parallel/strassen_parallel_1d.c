#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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
   for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         matrix[i*size+j] = (data_type)(rand() % 10);
      }
   }
}

void print_matrix(int size, data_type *matrix, const char *name) {
   printf("%s:\n", name);
   for (int i = 0; i < size; i++) {
       for (int j = 0; j < size; j++) {
           printf("%4d ", matrix[i * size + j]);
       }
       printf("\n");
   }
   printf("\n");
}

// Standard algorithm - multiplies A and B, C is output. This is borrowed from the lecture notes. 
void standard_matrix_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   memset(C, 0, size * size * sizeof(data_type));
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
   for (int base = 0; base < size; base++) {
      for(int i = 0; i < size; i++) {
            C[i + size*size*base] = A[i + size*size*base] + B[i + size*size*base];  
            printf("A:%d B:%d -- ", A[i + size*size*base], B[i + size*size*base]);
      }
   }
   printf("Size: %d\n", size);
}

void subtract_matrix(int size, data_type *A, data_type *B, data_type *C){
   #pragma omp parallel for
   for (int base = 0; base < size; base++) {
      for(int i = 0; i < size; i++) {
            C[i + size*size*base] = A[i + size*size*base] - B[i + size*size*base];  
            printf("A:%d B:%d -- ", A[i + size*size*base], B[i + size*size*base]);
      }
   }
   printf("Size: %d\n", size);
}

void strassen_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   if (size <= 2) { //Only uses strassen when the (sub)matrices are large
      
      {
         standard_matrix_multiplication(size, A, B, C, num_threads);
      }
      
      return;
   }
   int block_size = size / 2;

   data_type *a11 = A;
   data_type *a12 = A + block_size;
   data_type *a21 = A + block_size * size;
   data_type *a22 = A + block_size * size + block_size;

   print_matrix(block_size, a11, "a11");

   printf("a22: %d\n", *a22);

   data_type *b11 = B;
   data_type *b12 = B + block_size;
   data_type *b21 = B + block_size * size;
   data_type *b22 = B + block_size * size + block_size;

   printf("b22: %d\n", *b22);

   data_type *c11 = C;
   data_type *c12 = C + block_size;
   data_type *c21 = C + block_size * size;
   data_type *c22 = C + block_size * size + block_size;

   data_type *M1 = allocate_matrix(block_size);
   data_type *M2 = allocate_matrix(block_size);
   data_type *M3 = allocate_matrix(block_size);
   data_type *M4 = allocate_matrix(block_size);
   data_type *M5 = allocate_matrix(block_size);
   data_type *M6 = allocate_matrix(block_size);
   data_type *M7 = allocate_matrix(block_size);

   data_type *temp_result1 = allocate_matrix(block_size);
   data_type *temp_result2 = allocate_matrix(block_size);

   /* ---- Calculate M1 to M7 ---- */
   #pragma omp parallel num_threads(num_threads)
   {
      #pragma omp single nowait
      {
         // M1
         #pragma omp task private(temp_result1, temp_result2)
         {add_matrix(block_size, a11, a22, temp_result1);
         print_matrix(block_size, temp_result1, "a11 + a22");
         add_matrix(block_size, b11, b22, temp_result2);
         print_matrix(block_size, temp_result2, "b11 + b22");
         strassen_multiplication(block_size, temp_result1, temp_result2, M1, num_threads);}
         // M2
         #pragma omp task private(temp_result1)
         {add_matrix(block_size, a21, a22, temp_result1);
         strassen_multiplication(block_size, temp_result1, b11, M2, num_threads);}
         // M3
         #pragma omp task private(temp_result1)
         {subtract_matrix(block_size, b12, b22, temp_result1);
         strassen_multiplication(block_size, a11, temp_result1, M3, num_threads);}
         // M4
         #pragma omp task private(temp_result1)
         {subtract_matrix(block_size, b21, b11, temp_result1);
         strassen_multiplication(block_size, a22, temp_result1, M4, num_threads);}
         // M5
         #pragma omp task private(temp_result1)
         {add_matrix(block_size, a11, a12, temp_result1);
         strassen_multiplication(block_size, temp_result1, b22, M5, num_threads);}
         // M6
         #pragma omp task private(temp_result1, temp_result2)
         {subtract_matrix(block_size, a21, a11, temp_result1);
         add_matrix(block_size, b11, b12, temp_result2);
         strassen_multiplication(block_size, temp_result1, temp_result2, M6, num_threads);}
         // M7
         #pragma omp task private(temp_result1, temp_result2)
         {subtract_matrix(block_size, a12, a22, temp_result1);
         add_matrix(block_size, b21, b22, temp_result2);
         strassen_multiplication(block_size, temp_result1, temp_result2, M7, num_threads);}
         
         #pragma omp taskwait
      }
   }

   print_matrix(block_size, M1, "M1");
   print_matrix(block_size, M2, "M2");
   print_matrix(block_size, M3, "M3");
   print_matrix(block_size, M4, "M4");
   print_matrix(block_size, M5, "M5");
   print_matrix(block_size, M6, "M6");
   print_matrix(block_size, M7, "M7");

   /* ---- Calculate C11 to C22 ---- */

   #pragma omp parallel num_threads(4)
   {
      #pragma omp single nowait
      {
         //c11
         #pragma omp task private(temp_result1, temp_result2)
         {add_matrix(block_size, M1, M4, temp_result1);
         add_matrix(block_size, temp_result1, M7, temp_result2);
         subtract_matrix(block_size, temp_result2, M5, c11);}
         //c12
         #pragma omp task
         {add_matrix(block_size, M3, M5, c12);}
         //c21
         #pragma omp task
         {add_matrix(block_size, M2, M4, c21);}
         //c22
         #pragma omp task private(temp_result1, temp_result2)
         {add_matrix(block_size, M1, M3, temp_result1);
         add_matrix(block_size, temp_result1, M6, temp_result2);
         subtract_matrix(block_size, temp_result2, M2, c22);}

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

}

int main(int argc, char *argv[]) {
   if (argc != 3) {
      printf("Input should be: ./a.out [matrix size] [num_threads]\n");
      return -1;
   }
   srand(42);
   int size = atoi(argv[1]); //Input argument is matrix size
   int num_threads = atoi(argv[2]);

   data_type *A = allocate_matrix(size);
   data_type *B = allocate_matrix(size);
   data_type *C = allocate_matrix(size);
   data_type *C_check = allocate_matrix(size);

   fill_matrix(size, A);
   fill_matrix(size, B);

   print_matrix(size, A, "A");
   print_matrix(size, B, "B");

   double start_time = get_wall_seconds();

   strassen_multiplication(size, A, B, C, num_threads);

   double end_time = get_wall_seconds();
   printf("Time taken: %f seconds\n", end_time - start_time);

   // Used to check results. Compares a few entries in C with standard multiplication (by changing n_threshold)
   //printf("%d %d %d\n", C[1], C[60], C[431]);

   // Check C is correct.
   printf("Checking if matrix is correct...\n");
   standard_matrix_multiplication(size, A, B, C_check, num_threads); // Calculate standard result
   int correct = 1;
   for (int i = 0; i < size * size; i++) {
      if (C[i] != C_check[i]) {
         correct = 0;
         break;
      }
   }

   if (correct) {
      printf("Strassen multiplication result is correct.\n");
   } else {
      printf("Strassen multiplication result is incorrect.\n");
   }

   print_matrix(size, C, "C");
   print_matrix(size, C_check, "C_check");

   free(A);
   free(B);
   free(C);

   return 0;
}