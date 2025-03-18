#include <stdio.h>
#include <stdlib.h>
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

void add_matrix(int size, data_type *A, data_type *B, data_type *C) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] + B[i * size + j];
        }
    }
}

void subtract_matrix(int size, data_type *A, data_type *B, data_type *C) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] - B[i * size + j];
        }
    }
}

// Function to extract submatrix from a matrix
void extract_submatrix(int size, data_type *src, data_type *dst, int row_start, int col_start, int subsize) {
    for (int i = 0; i < subsize; i++) {
        for (int j = 0; j < subsize; j++) {
            dst[i * subsize + j] = src[(i + row_start) * size + (j + col_start)];
        }
    }
}

// Function to set a submatrix into a larger matrix
void set_submatrix(int size, data_type *dst, data_type *src, int row_start, int col_start, int subsize) {
    for (int i = 0; i < subsize; i++) {
        for (int j = 0; j < subsize; j++) {
            dst[(i + row_start) * size + (j + col_start)] = src[i * subsize + j];
        }
    }
}

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

void strassen_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
    // Initialize C to zeros
    memset(C, 0, size * size * sizeof(data_type));
    
    // Base case for recursion
    if (size <= 64) { // Threshold can be adjusted based on performance
        standard_matrix_multiplication(size, A, B, C, num_threads);
        return;
    }
    
    int block_size = size / 2;
    
    // Allocate memory for submatrices
    data_type *a11 = allocate_matrix(block_size);
    data_type *a12 = allocate_matrix(block_size);
    data_type *a21 = allocate_matrix(block_size);
    data_type *a22 = allocate_matrix(block_size);
    
    data_type *b11 = allocate_matrix(block_size);
    data_type *b12 = allocate_matrix(block_size);
    data_type *b21 = allocate_matrix(block_size);
    data_type *b22 = allocate_matrix(block_size);
    
    // Extract submatrices
    extract_submatrix(size, A, a11, 0, 0, block_size);
    extract_submatrix(size, A, a12, 0, block_size, block_size);
    extract_submatrix(size, A, a21, block_size, 0, block_size);
    extract_submatrix(size, A, a22, block_size, block_size, block_size);
    
    extract_submatrix(size, B, b11, 0, 0, block_size);
    extract_submatrix(size, B, b12, 0, block_size, block_size);
    extract_submatrix(size, B, b21, block_size, 0, block_size);
    extract_submatrix(size, B, b22, block_size, block_size, block_size);
    
    // Allocate memory for intermediate results
    data_type *M1 = allocate_matrix(block_size);
    data_type *M2 = allocate_matrix(block_size);
    data_type *M3 = allocate_matrix(block_size);
    data_type *M4 = allocate_matrix(block_size);
    data_type *M5 = allocate_matrix(block_size);
    data_type *M6 = allocate_matrix(block_size);
    data_type *M7 = allocate_matrix(block_size);
    
    data_type *temp1 = allocate_matrix(block_size);
    data_type *temp2 = allocate_matrix(block_size);
    
    #pragma omp parallel num_threads(num_threads)
    {
       #pragma omp single nowait
       {
          // M1
          #pragma omp task private(temp1, temp2)
          {add_matrix(block_size, a11, a22, temp1);
          print_matrix(block_size, temp1, "a11 + a22");
          add_matrix(block_size, b11, b22, temp2);
          print_matrix(block_size, temp2, "b11 + b22");
          strassen_multiplication(block_size, temp1, temp2, M1, num_threads);}
          // M2
          #pragma omp task private(temp1)
          {add_matrix(block_size, a21, a22, temp1);
          strassen_multiplication(block_size, temp1, b11, M2, num_threads);}
          // M3
          #pragma omp task private(temp1)
          {subtract_matrix(block_size, b12, b22, temp1);
          strassen_multiplication(block_size, a11, temp1, M3, num_threads);}
          // M4
          #pragma omp task private(temp1)
          {subtract_matrix(block_size, b21, b11, temp1);
          strassen_multiplication(block_size, a22, temp1, M4, num_threads);}
          // M5
          #pragma omp task private(temp1)
          {add_matrix(block_size, a11, a12, temp1);
          strassen_multiplication(block_size, temp1, b22, M5, num_threads);}
          // M6
          #pragma omp task private(temp1, temp2)
          {subtract_matrix(block_size, a21, a11, temp1);
          add_matrix(block_size, b11, b12, temp2);
          strassen_multiplication(block_size, temp1, temp2, M6, num_threads);}
          // M7
          #pragma omp task private(temp1, temp2)
          {subtract_matrix(block_size, a12, a22, temp1);
          add_matrix(block_size, b21, b22, temp2);
          strassen_multiplication(block_size, temp1, temp2, M7, num_threads);}
          
          #pragma omp taskwait
       }
    }
    // Calculate C submatrices
    data_type *c11 = allocate_matrix(block_size);
    data_type *c12 = allocate_matrix(block_size);
    data_type *c21 = allocate_matrix(block_size);
    data_type *c22 = allocate_matrix(block_size);
    
    // C11 = M1 + M4 - M5 + M7
    add_matrix(block_size, M1, M4, temp1);
    subtract_matrix(block_size, temp1, M5, temp2);
    add_matrix(block_size, temp2, M7, c11);
    
    // C12 = M3 + M5
    add_matrix(block_size, M3, M5, c12);
    
    // C21 = M2 + M4
    add_matrix(block_size, M2, M4, c21);
    
    // C22 = M1 - M2 + M3 + M6
    subtract_matrix(block_size, M1, M2, temp1);
    add_matrix(block_size, temp1, M3, temp2);
    add_matrix(block_size, temp2, M6, c22);
    
    // Set C submatrices in the result matrix
    set_submatrix(size, C, c11, 0, 0, block_size);
    set_submatrix(size, C, c12, 0, block_size, block_size);
    set_submatrix(size, C, c21, block_size, 0, block_size);
    set_submatrix(size, C, c22, block_size, block_size, block_size);
    
    // Free all allocated memory
    free(a11); free(a12); free(a21); free(a22);
    free(b11); free(b12); free(b21); free(b22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(temp1); free(temp2);
    free(c11); free(c12); free(c21); free(c22);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Input should be: ./a.out [matrix size] [num_threads]\n");
        return -1;
    }
    srand(42);
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    // Ensure size is a power of 2
    int power_of_two = 1;
    while (power_of_two < size) {
        power_of_two *= 2;
    }
    if (size != power_of_two) {
        printf("Warning: Adjusting matrix size to next power of 2: %d\n", power_of_two);
        size = power_of_two;
    }
    
    data_type *A = allocate_matrix(size);
    data_type *B = allocate_matrix(size);
    data_type *C = allocate_matrix(size);
    data_type *C_check = allocate_matrix(size);

    fill_matrix(size, A);
    fill_matrix(size, B);

    if (size <= 16) {
        print_matrix(size, A, "A");
        print_matrix(size, B, "B");
    } else {
        printf("Matrix size too large to print\n");
    }

    double start_time = get_wall_seconds();
    strassen_multiplication(size, A, B, C, num_threads);
    double end_time = get_wall_seconds();
    printf("Strassen multiplication time: %f seconds\n", end_time - start_time);

    printf("Verifying results with standard multiplication...\n");
    start_time = get_wall_seconds();
    standard_matrix_multiplication(size, A, B, C_check, num_threads);
    end_time = get_wall_seconds();
    printf("Standard multiplication time: %f seconds\n", end_time - start_time);

    int correct = 1;
    for (int i = 0; i < size * size; i++) {
        if (C[i] != C_check[i]) {
            correct = 0;
            printf("Mismatch at index %d: Strassen = %d, Standard = %d\n", 
                   i, C[i], C_check[i]);
            break;
        }
    }

    if (correct) {
        printf("Verification successful: Strassen multiplication result is correct.\n");
    } else {
        printf("Verification failed: Strassen multiplication result is incorrect.\n");
    }

    if (size <= 16) {
        print_matrix(size, C, "Result C");
    }

    free(A);
    free(B);
    free(C);
    free(C_check);

    return 0;
}