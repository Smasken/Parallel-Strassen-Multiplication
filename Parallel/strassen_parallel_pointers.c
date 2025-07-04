#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

// Timing function
static double get_wall_seconds() {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec + (double) tv.tv_usec / 1000000;
}

// Allocates memory for a size*size matrix
int *allocate_matrix(int size) {
   int *matrix = (int *)calloc(size*size, sizeof(int));
   return matrix;
}

void fill_matrix(int size, int *matrix) {
   for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         matrix[i*size+j] = (int)(rand() % 10);
      }
   }
}

void print_matrix(int size, int *matrix, const char *name, int stride) {
   printf("%s:\n", name);
   for (int i = 0; i < size; i++) {
       for (int j = 0; j < size; j++) {
           printf("%4d ", matrix[i * stride + j]);
       }
       printf("\n");
   }
   printf("\n");
}

// Standard algorithm - multiplies A and B, C is output. This is borrowed from the lecture notes. 
void standard_matrix_multiplication(int size, int *A, int *B, int *C) {
   for(int i = 0; i < size; i++) {
      for (int k = 0; k < size; k++) {
         for(int j = 0; j < size; j++) {
            C[i*size+j] += A[i*size+k] * B[k*size+j];
         }
      }
   }
}

void add_matrix(int mid, int *A, int *B, int *C, int stride) {
    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            C[i * mid + j] = A[i * stride + j] + B[i * stride + j];
        }
    }
}

void sub_matrix(int mid, int *A, int *B, int *C, int stride) {
    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            C[i * mid + j] = A[i * stride + j] - B[i * stride + j];
        }
    }
}

void strassen(int size, int *A, int *B, int *C, int depth, int max_depth) {
   if (size == 2 || depth > max_depth) {
        standard_matrix_multiplication(size, A, B, C);
        return;
    }

   int mid = size / 2;
   int block = block;

   int *a11 = A;
   int *a12 = A + mid;
   int *a21 = A + mid * size;
   int *a22 = A + mid * size + mid;

   print_matrix(mid, a11, "a11", size);

   int *b11 = B;
   int *b12 = B + mid;
   int *b21 = B + mid * size;
   int *b22 = B + mid * size + mid;

   int *c11 = C;
   int *c12 = C + mid;
   int *c21 = C + mid * size;
   int *c22 = C + mid * size + mid;

   int *buffer = (int *)calloc(18 * block, sizeof(int));

   int *M1 = buffer;
   int *M2 = buffer + 1 * block;
   int *M3 = buffer + 2 * block;
   int *M4 = buffer + 3 * block;
   int *M5 = buffer + 4 * block;
   int *M6 = buffer + 5 * block;
   int *M7 = buffer + 6 * block;

   int *temp_result1  = buffer + 7 * block;
   int *temp_result2  = buffer + 8 * block;
   int *temp_result3  = buffer + 9 * block;
   int *temp_result4  = buffer + 10 * block;
   int *temp_result5  = buffer + 11 * block;
   int *temp_result6  = buffer + 12 * block;
   int *temp_result7  = buffer + 13 * block;
   int *temp_result8  = buffer + 14 * block;
   int *temp_result9  = buffer + 15 * block;
   int *temp_result10 = buffer + 16 * block;

   {
   /* ---- Calculate M1 to M7 ---- */
   // M1
   #pragma omp task
   {add_matrix(mid, a11, a22, temp_result1, size);
   print_matrix(mid, temp_result1, "temp1", mid);
   add_matrix(mid, b11, b22, temp_result2, size);
   print_matrix(mid, temp_result2, "temp2", mid);
   strassen(mid, temp_result1, temp_result2, M1, depth +1, max_depth);}
   print_matrix(mid, M1, "M1", mid);
   // M2
   #pragma omp task
   {add_matrix(mid, a21, a22, temp_result3, size);
   strassen(mid, temp_result3, b11, M2, depth +1, max_depth);}
   // M3
   #pragma omp task
   {sub_matrix(mid, b12, b22, temp_result4, size);
   strassen(mid, a11, temp_result4, M3, depth +1, max_depth);}
   // M4
   #pragma omp task
   {sub_matrix(mid, b21, b11, temp_result5, size);
   strassen(mid, a22, temp_result5, M4, depth +1, max_depth);}
   // M5
   #pragma omp task
   {add_matrix(mid, a11, a12, temp_result6, size);
   strassen(mid, temp_result6, b22, M5, depth +1, max_depth);}
   // M6
   #pragma omp task
   {sub_matrix(mid, a21, a11, temp_result7, size);
   add_matrix(mid, b11, b12, temp_result8, size);
   strassen(mid, temp_result7, temp_result8, M6, depth +1, max_depth);}
   // M7
   #pragma omp task
   {sub_matrix(mid, a12, a22, temp_result9, size);
   add_matrix(mid, b21, b22, temp_result10, size);
   strassen(mid, temp_result9, temp_result10, M7, depth +1, max_depth);}

   #pragma omp taskwait

   /* ---- Calculate C11 to C22 ---- */
   // C11 = M1 + M4 + M7 - M5
   #pragma omp task
   for (int i = 0; i < mid; i++)
      for (int j = 0; j < mid; j++)
         c11[i * size + j] = M1[i * mid + j] + M4[i * mid + j] + M7[i * mid + j] - M5[i * mid + j];

   // C12 = M3 + M5
   #pragma omp task
   for (int i = 0; i < mid; i++)
      for (int j = 0; j < mid; j++)
         c12[i * size + j] = M3[i * mid + j] + M5[i * mid + j];

   // C21 = M2 + M4
   #pragma omp task
   for (int i = 0; i < mid; i++)
      for (int j = 0; j < mid; j++)
         c21[i * size + j] = M2[i * mid + j] + M4[i * mid + j];
   // C22 = M1 + M3 + M6 - M2
   #pragma omp task
   for (int i = 0; i < mid; i++)
      for (int j = 0; j < mid; j++)
         c22[i * size + j] = M1[i * mid + j] + M3[i * mid + j] + M6[i * mid + j] - M2[i * mid + j];
   }
   free(buffer);
   }

int main(int argc, char *argv[]) {
   if (argc != 4) {
      printf("Input should be: ./a.out [matrix size] [num_threads] [recursion depth]\n");
      return -1;
   }
   srand(42);
   int size = atoi(argv[1]); //Input argument is matrix size
   int num_threads = atoi(argv[2]);
   int max_depth = atoi(argv[3]);
   int depth = 0;

   int *A = allocate_matrix(size);
   int *B = allocate_matrix(size);
   int *C = allocate_matrix(size);
   int *C_check = allocate_matrix(size);

   fill_matrix(size, A);
   fill_matrix(size, B);

   double start_time = get_wall_seconds();
   omp_set_num_threads(num_threads);
   #pragma omp parallel
      {
         #pragma omp single
         strassen(size, A, B, C, depth, max_depth);
      }

   double end_time = get_wall_seconds();
   printf("Time taken: %f seconds\n", end_time - start_time);

   // Check C is correct.
   printf("Checking if matrix is correct...\n");
   standard_matrix_multiplication(size, A, B, C_check); // Calculate standard result
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
   if (size <=8)
   {
   print_matrix(size, A, "A", size);
   print_matrix(size, B, "B", size);
   print_matrix(size, C, "Strassen", size);
   print_matrix(size, C_check, "Standard", size);
   }

   free(A);
   free(B);
   free(C);
   free(C_check);

   return 0;
}