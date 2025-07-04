#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

static double get_wall_seconds() {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec + (double) tv.tv_usec / 1000000;
}

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

void add_matrix(int block_size, int *A, int *B, int *C, int stride) {
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            C[i * block_size + j] = A[i * stride + j] + B[i * stride + j];
        }
    }
}

void sub_matrix(int block_size, int *A, int *B, int *C, int stride) {
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            C[i * block_size + j] = A[i * stride + j] - B[i * stride + j];
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

void standard_matrix_multiplication(int size, int *A, int *B, int *C) {
    for (int i = 0; i < size * size; i++) {
        C[i] = 0;
    }
    for(int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for(int j = 0; j < size; j++) {
            C[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }
}

void strassen(int size, int *A, int *B, int *C, int depth, int max_depth) {
   if (size == 2 || depth > max_depth) {
        standard_matrix_multiplication(size, A, B, C);
        return;
    }

    int block_size = size / 2;

    int *a11 = A;
    int *a12 = A + block_size;
    int *a21 = A + block_size * size;
    int *a22 = A + block_size * size + block_size;

    int *b11 = B;
    int *b12 = B + block_size;
    int *b21 = B + block_size * size;
    int *b22 = B + block_size * size + block_size;

    int *c11 = C;
    int *c12 = C + block_size;
    int *c21 = C + block_size * size;
    int *c22 = C + block_size * size + block_size;

    int *buffer = (int *)calloc(17 * block_size * block_size, sizeof(int));

    int *M1 = buffer;
    int *M2 = buffer + 1 * block_size * block_size;
    int *M3 = buffer + 2 * block_size * block_size;
    int *M4 = buffer + 3 * block_size * block_size;
    int *M5 = buffer + 4 * block_size * block_size;
    int *M6 = buffer + 5 * block_size * block_size;
    int *M7 = buffer + 6 * block_size * block_size;

    int *temp_result1 = buffer + 7 * block_size * block_size;
    int *temp_result2 = buffer + 8 * block_size * block_size;
    int *temp_result3 = buffer + 9 * block_size * block_size;
    int *temp_result4 = buffer + 10 * block_size * block_size;
    int *temp_result5 = buffer + 11 * block_size * block_size;
    int *temp_result6 = buffer + 12 * block_size * block_size;
    int *temp_result7 = buffer + 13 * block_size * block_size;
    int *temp_result8 = buffer + 14 * block_size * block_size;
    int *temp_result9 = buffer + 15 * block_size * block_size;
    int *temp_result10 = buffer + 16 * block_size * block_size;

    // M1
    add_matrix(block_size, a11, a22, temp_result1, size);
    add_matrix(block_size, b11, b22, temp_result2, size);
    strassen(block_size, temp_result1, temp_result2, M1, depth +1, max_depth);
    print_matrix(block_size, M1, "M1", block_size);
    // M2
    {add_matrix(block_size, a21, a22, temp_result3, size);
    strassen(block_size, temp_result3, b11, M2, depth +1, max_depth);}
    // M3
    {sub_matrix(block_size, b12, b22, temp_result4, size);
    strassen(block_size, a11, temp_result4, M3, depth +1, max_depth);}
    // M4
    {sub_matrix(block_size, b21, b11, temp_result5, size);
    strassen(block_size, a22, temp_result5, M4, depth +1, max_depth);}
    print_matrix(block_size, M4, "M4", block_size);
    // M5
    {add_matrix(block_size, a11, a12, temp_result6, size);
    strassen(block_size, temp_result6, b22, M5, depth +1, max_depth);}
    print_matrix(block_size, M5, "M5", block_size);
    // M6
    {sub_matrix(block_size, a21, a11, temp_result7, size);
    add_matrix(block_size, b11, b12, temp_result8, size);
    strassen(block_size, temp_result7, temp_result8, M6, depth +1, max_depth);}
    // M7
    {sub_matrix(block_size, a12, a22, temp_result9, size);
    add_matrix(block_size, b21, b22, temp_result10, size);
    strassen(block_size, temp_result9, temp_result10, M7, depth +1, max_depth);}
    print_matrix(block_size, M7, "M5", block_size);

    /* ---- Calculate C11 to C22 ---- */
    // C11 = M1 + M4 + M7 - M5
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            c11[i * size + j] = M1[i * block_size + j] + M4[i * block_size + j] + M7[i * block_size + j] - M5[i * block_size + j];

    // C12 = M3 + M5
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            c12[i * size + j] = M3[i * block_size + j] + M5[i * block_size + j];

    // C21 = M2 + M4
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            c21[i * size + j] = M2[i * block_size + j] + M4[i * block_size + j];
    // C22 = M1 + M3 + M6 - M2
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            c22[i * size + j] = M1[i * block_size + j] + M3[i * block_size + j] + M6[i * block_size + j] - M2[i * block_size + j];

    printf("Matrix size: %d, block_size: %d\n", size, block_size);

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
  
    strassen(size, A, B, C, depth, max_depth);


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