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
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
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
    #pragma omp parallel for if(size > 64)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] + B[i * size + j];
        }
    }
}

void subtract_matrix(int size, data_type *A, data_type *B, data_type *C) {
    #pragma omp parallel for if(size > 64)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] - B[i * size + j];
        }
    }
}

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

void strassen_multiplication(int size, data_type *A, data_type *B, data_type *C, int num_threads) {
    // Base case: use standard multiplication for small matrices
    if (size <= 128) {
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
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            a11[i * block_size + j] = A[i * size + j];
            a12[i * block_size + j] = A[i * size + (j + block_size)];
            a21[i * block_size + j] = A[(i + block_size) * size + j];
            a22[i * block_size + j] = A[(i + block_size) * size + (j + block_size)];

            b11[i * block_size + j] = B[i * size + j];
            b12[i * block_size + j] = B[i * size + (j + block_size)];
            b21[i * block_size + j] = B[(i + block_size) * size + j];
            b22[i * block_size + j] = B[(i + block_size) * size + (j + block_size)];
        }
    }
    
    // Calculate C submatrices
    data_type *c11 = allocate_matrix(block_size);
    data_type *c12 = allocate_matrix(block_size);
    data_type *c21 = allocate_matrix(block_size);
    data_type *c22 = allocate_matrix(block_size);
    
    // Allocate memory for intermediate results
    data_type *M1 = allocate_matrix(block_size);
    data_type *M2 = allocate_matrix(block_size);
    data_type *M3 = allocate_matrix(block_size);
    data_type *M4 = allocate_matrix(block_size);
    data_type *M5 = allocate_matrix(block_size);
    data_type *M6 = allocate_matrix(block_size);
    data_type *M7 = allocate_matrix(block_size);

    #pragma omp parallel num_threads(num_threads)
    {
        data_type *temp1 = allocate_matrix(block_size); //allocate temps inside of thread to avoid race conditions
        data_type *temp2 = allocate_matrix(block_size);

        #pragma omp sections
        {
            #pragma omp section
            {
                // M1 = (A11 + A22) * (B11 + B22)
                add_matrix(block_size, a11, a22, temp1);
                add_matrix(block_size, b11, b22, temp2);
                strassen_multiplication(block_size, temp1, temp2, M1, num_threads);
            }

            #pragma omp section
            {
                // M2 = (A21 + A22) * B11
                add_matrix(block_size, a21, a22, temp1);
                strassen_multiplication(block_size, temp1, b11, M2, num_threads);
            }

            #pragma omp section
            {
                // M3 = A11 * (B12 - B22)
                subtract_matrix(block_size, b12, b22, temp2);
                strassen_multiplication(block_size, a11, temp2, M3, num_threads);
            }

            #pragma omp section
            {
                // M4 = A22 * (B21 - B11)
                subtract_matrix(block_size, b21, b11, temp2);
                strassen_multiplication(block_size, a22, temp2, M4, num_threads);
            }

            #pragma omp section
            {
                // M5 = (A11 + A12) * B22
                add_matrix(block_size, a11, a12, temp1);
                strassen_multiplication(block_size, temp1, b22, M5, num_threads);
            }

            #pragma omp section
            {
                // M6 = (A21 - A11) * (B11 + B12)
                subtract_matrix(block_size, a21, a11, temp1);
                add_matrix(block_size, b11, b12, temp2);
                strassen_multiplication(block_size, temp1, temp2, M6, num_threads);
            }

            #pragma omp section
            {
                // M7 = (A12 - A22) * (B21 + B22)
                subtract_matrix(block_size, a12, a22, temp1);
                add_matrix(block_size, b21, b22, temp2);
                strassen_multiplication(block_size, temp1, temp2, M7, num_threads);
            }
        }

        free(temp1);
        free(temp2);
    }

    // C11 = M1 + M4 - M5 + M7
    add_matrix(block_size, M1, M4, c11);
    subtract_matrix(block_size, c11, M5, c11);
    add_matrix(block_size, c11, M7, c11);

    // C12 = M3 + M5
    add_matrix(block_size, M3, M5, c12);

    // C21 = M2 + M4
    add_matrix(block_size, M2, M4, c21);

    // C22 = M1 - M2 + M3 + M6
    subtract_matrix(block_size, M1, M2, c22);
    add_matrix(block_size, c22, M3, c22);
    add_matrix(block_size, c22, M6, c22);
    
    // Set C submatrices in the result matrix
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            C[i * size + j] = c11[i * block_size + j];
            C[i * size + (j + block_size)] = c12[i * block_size + j];
            C[(i + block_size) * size + j] = c21[i * block_size + j];
            C[(i + block_size) * size + (j + block_size)] = c22[i * block_size + j];
        }
    }
    
    // Free all allocated memory
    free(a11); free(a12); free(a21); free(a22);
    free(b11); free(b12); free(b21); free(b22);
    free(c11); free(c12); free(c21); free(c22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Input should be: ./a.out [matrix size] [num_threads]\n");
        return -1;
    }
    //srand(42); Uncomment this to get consistent matrices (better for bug checking)
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    data_type *A = allocate_matrix(size);
    data_type *B = allocate_matrix(size);
    data_type *C = allocate_matrix(size);
    data_type *C_check = allocate_matrix(size);

    fill_matrix(size, A);
    fill_matrix(size, B);

    double start_time = get_wall_seconds();
    strassen_multiplication(size, A, B, C, num_threads);
    double end_time = get_wall_seconds();
    printf("Strassen multiplication time: %f seconds\n", end_time - start_time);

    // Below checks if the result is correct
    printf("Verifying results with standard multiplication...\n");
    start_time = get_wall_seconds();
    standard_matrix_multiplication(size, A, B, C_check, num_threads);
    end_time = get_wall_seconds();
    printf("Standard multiplication time: %f seconds\n", end_time - start_time);
    
    printf("Checking if matrix is correct...\n");
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

    if (size < 8) {
    print_matrix(size, C, "C");
    print_matrix(size, C_check, "C_check");
    }

    free(A);
    free(B);
    free(C);
    free(C_check);

    return 0;
}