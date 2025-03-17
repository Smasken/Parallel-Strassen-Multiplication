#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

// This allows changing what type to fill the matrix with (int, float, etc.)
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
   if (matrix == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
   }
   return matrix;
}

// Fills matrix with random values
void fill_matrix(int size, data_type *matrix) {
   for (int i = 0; i < size*size; i++) {
      matrix[i] = (data_type)(rand() % 10);
   }
}

// Standard matrix multiplication algorithm
void standard_matrix_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   // Initialize C to zero
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

// Add matrices: C = A + B
void add_matrices(int size, data_type *A, data_type *B, data_type *C) {
   for(int i = 0; i < size*size; i++) {
      C[i] = A[i] + B[i];
   }
}

// Subtract matrices: C = A - B
void subtract_matrices(int size, data_type *A, data_type *B, data_type *C) {
   for(int i = 0; i < size*size; i++) {
      C[i] = A[i] - B[i];
   }
}

// Split parent matrix into 4 child matrices
void split_matrix(int parent_size, data_type *parent, int child_size, 
                 data_type *child11, data_type *child12, 
                 data_type *child21, data_type *child22) {
   for (int i = 0; i < child_size; i++) {
      for (int j = 0; j < child_size; j++) {
         // Top-left (child11)
         child11[i*child_size + j] = parent[i*parent_size + j];
         
         // Top-right (child12)
         child12[i*child_size + j] = parent[i*parent_size + j + child_size];
         
         // Bottom-left (child21)
         child21[i*child_size + j] = parent[(i + child_size)*parent_size + j];
         
         // Bottom-right (child22)
         child22[i*child_size + j] = parent[(i + child_size)*parent_size + j + child_size];
      }
   }
}

// Merge 4 child matrices into a parent matrix
void merge_matrix(int parent_size, data_type *parent, int child_size, 
                 data_type *child11, data_type *child12, 
                 data_type *child21, data_type *child22) {
   for (int i = 0; i < child_size; i++) {
      for (int j = 0; j < child_size; j++) {
         // Top-left (child11)
         parent[i*parent_size + j] = child11[i*child_size + j];
         
         // Top-right (child12)
         parent[i*parent_size + j + child_size] = child12[i*child_size + j];
         
         // Bottom-left (child21)
         parent[(i + child_size)*parent_size + j] = child21[i*child_size + j];
         
         // Bottom-right (child22)
         parent[(i + child_size)*parent_size + j + child_size] = child22[i*child_size + j];
      }
   }
}

// Strassen algorithm for matrix multiplication
void strassen_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
   // For small matrices, use standard multiplication
   if (size <= 256) {  // Increased threshold for better performance
      standard_matrix_multiplication(size, A, B, C, num_threads);
      return;
   }
   
   int new_size = size / 2;
   
   // Allocate memory for submatrices
   data_type *a11 = allocate_matrix(new_size);
   data_type *a12 = allocate_matrix(new_size);
   data_type *a21 = allocate_matrix(new_size);
   data_type *a22 = allocate_matrix(new_size);
   
   data_type *b11 = allocate_matrix(new_size);
   data_type *b12 = allocate_matrix(new_size);
   data_type *b21 = allocate_matrix(new_size);
   data_type *b22 = allocate_matrix(new_size);
   
   data_type *c11 = allocate_matrix(new_size);
   data_type *c12 = allocate_matrix(new_size);
   data_type *c21 = allocate_matrix(new_size);
   data_type *c22 = allocate_matrix(new_size);
   
   // Split matrices A and B into 4 submatrices each
   split_matrix(size, A, new_size, a11, a12, a21, a22);
   split_matrix(size, B, new_size, b11, b12, b21, b22);
   
   // Allocate memory for intermediate matrices
   data_type *m1 = allocate_matrix(new_size);
   data_type *m2 = allocate_matrix(new_size);
   data_type *m3 = allocate_matrix(new_size);
   data_type *m4 = allocate_matrix(new_size);
   data_type *m5 = allocate_matrix(new_size);
   data_type *m6 = allocate_matrix(new_size);
   data_type *m7 = allocate_matrix(new_size);
   
   data_type *temp1 = allocate_matrix(new_size);
   data_type *temp2 = allocate_matrix(new_size);
   
   // Calculate M1 = (A11 + A22) * (B11 + B22)
   add_matrices(new_size, a11, a22, temp1);
   add_matrices(new_size, b11, b22, temp2);
   strassen_multiplication(new_size, temp1, temp2, m1, num_threads);
   
   // Calculate M2 = (A21 + A22) * B11
   add_matrices(new_size, a21, a22, temp1);
   strassen_multiplication(new_size, temp1, b11, m2, num_threads);
   
   // Calculate M3 = A11 * (B12 - B22)
   subtract_matrices(new_size, b12, b22, temp1);
   strassen_multiplication(new_size, a11, temp1, m3, num_threads);
   
   // Calculate M4 = A22 * (B21 - B11)
   subtract_matrices(new_size, b21, b11, temp1);
   strassen_multiplication(new_size, a22, temp1, m4, num_threads);
   
   // Calculate M5 = (A11 + A12) * B22
   add_matrices(new_size, a11, a12, temp1);
   strassen_multiplication(new_size, temp1, b22, m5, num_threads);
   
   // Calculate M6 = (A21 - A11) * (B11 + B12)
   subtract_matrices(new_size, a21, a11, temp1);
   add_matrices(new_size, b11, b12, temp2);
   strassen_multiplication(new_size, temp1, temp2, m6, num_threads);
   
   // Calculate M7 = (A12 - A22) * (B21 + B22)
   subtract_matrices(new_size, a12, a22, temp1);
   add_matrices(new_size, b21, b22, temp2);
   strassen_multiplication(new_size, temp1, temp2, m7, num_threads);
   
   // Calculate C11 = M1 + M4 - M5 + M7
   add_matrices(new_size, m1, m4, temp1);
   subtract_matrices(new_size, temp1, m5, temp2);
   add_matrices(new_size, temp2, m7, c11);
   
   // Calculate C12 = M3 + M5
   add_matrices(new_size, m3, m5, c12);
   
   // Calculate C21 = M2 + M4
   add_matrices(new_size, m2, m4, c21);
   
   // Calculate C22 = M1 - M2 + M3 + M6
   subtract_matrices(new_size, m1, m2, temp1);
   add_matrices(new_size, temp1, m3, temp2);
   add_matrices(new_size, temp2, m6, c22);
   
   // Merge the submatrices into the result matrix
   merge_matrix(size, C, new_size, c11, c12, c21, c22);
   
   // Free memory
   free(a11); free(a12); free(a21); free(a22);
   free(b11); free(b12); free(b21); free(b22);
   free(c11); free(c12); free(c21); free(c22);
   free(m1); free(m2); free(m3); free(m4); free(m5); free(m6); free(m7);
   free(temp1); free(temp2);
}

int main(int argc, char *argv[]) {
   if (argc != 3) {
      printf("Usage: %s [matrix size] [num_threads]\n", argv[0]);
      return -1;
   }

   int size = atoi(argv[1]); // Input argument is matrix size
   int num_threads = atoi(argv[2]);
   
   // Ensure size is a power of 2
   if ((size & (size - 1)) != 0) {
      printf("Matrix size must be a power of 2\n");
      return -1;
   }

   // Allocate and initialize matrices
   data_type *A = allocate_matrix(size);
   data_type *B = allocate_matrix(size);
   data_type *C = allocate_matrix(size);
   data_type *C_standard = allocate_matrix(size);

   // Initialize matrices with random values
   srand(42);  // Set seed for reproducible results
   fill_matrix(size, A);
   fill_matrix(size, B);

   // Run Strassen's algorithm and time it
   printf("Running Strassen's algorithm for %dx%d matrices with %d threads...\n", size, size, num_threads);
   double start_time = get_wall_seconds();
   strassen_multiplication(size, A, B, C, num_threads);
   double end_time = get_wall_seconds();
   printf("Time taken for Strassen: %f seconds\n", end_time - start_time);

   // Run standard multiplication for comparison
   printf("Running standard multiplication for verification...\n");
   start_time = get_wall_seconds();
   standard_matrix_multiplication(size, A, B, C_standard, num_threads);
   end_time = get_wall_seconds();
   printf("Time taken for standard multiplication: %f seconds\n", end_time - start_time);

   // Verify results
   printf("Verifying results...\n");
   int correct = 1;
   for (int i = 0; i < size * size; i++) {
      if (C[i] != C_standard[i]) {
         printf("Mismatch at element %d: Strassen = %d, Standard = %d\n", 
                i, C[i], C_standard[i]);
         correct = 0;
         // Only show up to 5 mismatches
         if (i >= 4) break;
      }
   }

   if (correct) {
      printf("Verification successful: Strassen multiplication result is correct.\n");
   } else {
      printf("Verification failed: Strassen multiplication result is incorrect.\n");
   }

   // Clean up
   free(A);
   free(B);
   free(C);
   free(C_standard);

   return 0;
}