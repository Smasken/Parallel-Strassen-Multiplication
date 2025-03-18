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

void print_submatrix(int size, data_type *matrix, const char *name) {
    printf("%s:\n", name);
    for (int j = 0; j < size; j++) {
        for(int i = 0; i < size; i++) { 
            printf("%4d ", matrix[i + size*size*j]);
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

void add_submatrix(int size, data_type *A, data_type *B, data_type *C){
    //#pragma omp parallel for
    for (int j = 0; j < size; j++) {
       for(int i = 0; i < size; i++) {
             C[i + size*size*j] = A[i + size*size*j] + B[i + size*size*j];
       }
    }
}

void subtract_matrix(int size, data_type *A, data_type *B, data_type *C) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = A[i * size + j] - B[i * size + j];
            printf("A:%d B:%d -- ", A[i * size + j], B[i * size + j]);
        }
    }
}

void subtract_submatrix(int size, data_type *A, data_type *B, data_type *C){
    //#pragma omp parallel for
    for (int j = 0; j < size; j++) {
       for(int i = 0; i < size; i++) {
             C[i + size*size*j] = A[i + size*size*j] - B[i + size*size*j];
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
    if (size <= 2) { //Only uses strassen when the submatrices are large
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

    data_type *b11 = B;
    data_type *b12 = B + block_size;
    data_type *b21 = B + block_size * size;
    data_type *b22 = B + block_size * size + block_size;

    print_submatrix(block_size, a11, "a11");
    print_submatrix(block_size, b11, "b11");

    data_type *M1 = allocate_matrix(block_size);
    data_type *M2 = allocate_matrix(block_size);
    data_type *M3 = allocate_matrix(block_size);
    data_type *M4 = allocate_matrix(block_size);
    data_type *M5 = allocate_matrix(block_size);
    data_type *M6 = allocate_matrix(block_size);
    data_type *M7 = allocate_matrix(block_size);

    data_type *temp_result1 = allocate_matrix(block_size);
    data_type *temp_result2 = allocate_matrix(block_size);

    //------------ M1 - M7 --------------
    //M1
    add_submatrix(block_size, a11, a22, temp_result1);
    add_submatrix(block_size, b11, b22, temp_result2);
    //strassen_multiplication(block_size, temp_result1, temp_result2, M1, num_threads);
    print_submatrix(block_size, temp_result1, "a11 + a22");
    print_submatrix(block_size, temp_result2, "b11 + b22");
    
    standard_matrix_multiplication(block_size, temp_result1, temp_result2, M1, num_threads);
    print_submatrix(block_size, M1, "M1");

    free(M1);
    free(M2);
    free(temp_result1);
    free(temp_result2);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Input should be: ./a.out [matrix size] [num_threads]\n");
        return -1;
    }
    srand(42);
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    data_type *A = allocate_matrix(size);
    data_type *B = allocate_matrix(size);
    data_type *C = allocate_matrix(size);
    data_type *C_check = allocate_matrix(size);

    fill_matrix(size, A);
    fill_matrix(size, B);

    print_matrix(size, A, "A");
    print_matrix(size, B, "B");

    strassen_multiplication(size, A, B, C, num_threads);

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

   free(A);
   free(B);
   free(C);

   return 0;
}