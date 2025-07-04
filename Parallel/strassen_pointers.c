#include <stdio.h>
#include <stdlib.h>
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

void standard_matrix_multiplication(int size, int *A, int *B, int *C) {
    for(int i = 0; i < size; i++) {
       for (int k = 0; k < size; k++) {
          for(int j = 0; j < size; j++) {
             C[i*size+j] += A[i*size+k] * B[k*size+j];
          }
       }
    }
}

void print_matrix(const int* M, int N) {
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            printf("%4d ", M[y * N + x]);
        }
        printf("\n");
    }
}

// Extract submatrix from larger matrix, starting at a pointer representing the quadrant
void get_submatrix(const int* src, int* dst, int src_row, int src_col, int src_size, int sub_size) {
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            dst[i * sub_size + j] = src[(src_row + i) * src_size + (src_col + j)];
        }
    }
}

// Add submatrix result back to destination matrix
void add_to_submatrix(int* dst, const int* src, int dst_row, int dst_col, int dst_size, int sub_size) {
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            dst[(dst_row + i) * dst_size + (dst_col + j)] += src[i * sub_size + j];
        }
    }
}

void add_matrix(const int* A, const int* B, int* C, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] + B[i];
    }
}

void sub_matrix(const int* A, const int* B, int* C, int size) {
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] - B[i];
    }
}

// Recursive Strassen matrix multiplication with row-major storage
void strassen(int* A, int* B, int* C, int size, int depth, int max_depth, int* temp) {
    if (size == 2 || depth >= max_depth) {
        standard_matrix_multiplication(size, A, B, C);
        return;
    }

    int half = size / 2;
    int block = half * half;

    // Allocate temporary submatrices from temp buffer
    int* A11 = temp;
    int* A12 = temp + block;
    int* A21 = temp + 2 * block;
    int* A22 = temp + 3 * block;
    int* B11 = temp + 4 * block;
    int* B12 = temp + 5 * block;
    int* B21 = temp + 6 * block;
    int* B22 = temp + 7 * block;
    
    int* T1 = temp + 8 * block;
    int* T2 = temp + 9 * block;
    int* M1 = temp + 10 * block;
    int* M2 = temp + 11 * block;
    int* M3 = temp + 12 * block;
    int* M4 = temp + 13 * block;
    int* M5 = temp + 14 * block;
    int* M6 = temp + 15 * block;
    int* M7 = temp + 16 * block;

    // Extract submatrices from A
    get_submatrix(A, A11, 0, 0, size, half);        // top-left
    get_submatrix(A, A12, 0, half, size, half);     // top-right
    get_submatrix(A, A21, half, 0, size, half);     // bottom-left
    get_submatrix(A, A22, half, half, size, half);  // bottom-right

    // Extract submatrices from B
    get_submatrix(B, B11, 0, 0, size, half);        // top-left
    get_submatrix(B, B12, 0, half, size, half);     // top-right
    get_submatrix(B, B21, half, 0, size, half);     // bottom-left
    get_submatrix(B, B22, half, half, size, half);  // bottom-right

    // Initialize M matrices to zero
    memset(M1, 0, 7 * block * sizeof(int));

    // M1 = (A11 + A22) * (B11 + B22)
    add_matrix(A11, A22, T1, half);
    add_matrix(B11, B22, T2, half);
    strassen(T1, T2, M1, half, depth + 1, max_depth, temp + 17 * block);

    // M2 = (A21 + A22) * B11
    add_matrix(A21, A22, T1, half);
    strassen(T1, B11, M2, half, depth + 1, max_depth, temp + 17 * block);

    // M3 = A11 * (B12 - B22)
    sub_matrix(B12, B22, T1, half);
    strassen(A11, T1, M3, half, depth + 1, max_depth, temp + 17 * block);

    // M4 = A22 * (B21 - B11)
    sub_matrix(B21, B11, T1, half);
    strassen(A22, T1, M4, half, depth + 1, max_depth, temp + 17 * block);

    // M5 = (A11 + A12) * B22
    add_matrix(A11, A12, T1, half);
    strassen(T1, B22, M5, half, depth + 1, max_depth, temp + 17 * block);

    // M6 = (A21 - A11) * (B11 + B12)
    sub_matrix(A21, A11, T1, half);
    add_matrix(B11, B12, T2, half);
    strassen(T1, T2, M6, half, depth + 1, max_depth, temp + 17 * block);

    // M7 = (A12 - A22) * (B21 + B22)  
    sub_matrix(A12, A22, T1, half);
    add_matrix(B21, B22, T2, half);
    strassen(T1, T2, M7, half, depth + 1, max_depth, temp + 17 * block);

    // Compute result submatrices and add back to C
    // C11 = M1 + M4 - M5 + M7
    for (int i = 0; i < block; i++)
        T1[i] = M1[i] + M4[i] - M5[i] + M7[i];
    add_to_submatrix(C, T1, 0, 0, size, half);

    // C12 = M3 + M5
    for (int i = 0; i < block; i++)
        T1[i] = M3[i] + M5[i];
    add_to_submatrix(C, T1, 0, half, size, half);

    // C21 = M2 + M4
    for (int i = 0; i < block; i++)
        T1[i] = M2[i] + M4[i];
    add_to_submatrix(C, T1, half, 0, size, half);

    // C22 = M1 - M2 + M3 + M6
    for (int i = 0; i < block; i++)
        T1[i] = M1[i] - M2[i] + M3[i] + M6[i];
    add_to_submatrix(C, T1, half, half, size, half);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Input should be: ./a.out [matrix size] [recursion_depth]\n");
        return -1;
    }

    int N = atoi(argv[1]);
    int max_depth = atoi(argv[2]);
    int size = N * N;
    int* A = calloc(size, sizeof(int));
    int* B = calloc(size, sizeof(int));
    int* C = calloc(size, sizeof(int));
    int* C_check = calloc(size, sizeof(int));
    int* temp = calloc(size * 17, sizeof(int));

    // Fill A and B with simple values for demo
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            A[y * N + x] = x + y * N;
            B[y * N + x] = (x + y * N) % 5;
        }
    }
    if (N <= 8) {
    printf("Matrix A:\n");
    print_matrix(A, N);
    printf("\nMatrix B:\n");
    print_matrix(B, N);
    }

    double start_time = get_wall_seconds();
    strassen(A, B, C, N, 0, max_depth, temp);
    double end_time = get_wall_seconds();
    printf("Strassen multiplication time: %f seconds\n", end_time - start_time);

    // Below checks if the result is correct
    printf("Verifying results with standard multiplication...\n");
    start_time = get_wall_seconds();
    standard_matrix_multiplication(N, A, B, C_check);
    end_time = get_wall_seconds();
    printf("Standard multiplication time: %f seconds\n", end_time - start_time);
    
    printf("Checking if matrix is correct...\n");
    int correct = 1;
    for (int i = 0; i < size; i++) {
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

    if (N <= 8) {
        printf("\nMatrix C = A * B:\n");
        print_matrix(C, N);
        printf("\n");
        print_matrix(C_check, N);
        }

    free(A);
    free(B);
    free(C);
    free(C_check);
    free(temp);
    return 0;
}