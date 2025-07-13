#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
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

void print_matrix(const int* matrix, int size) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            printf("%4d ", matrix[y * size + x]);
        }
        printf("\n");
    }
}

// Extract submatrix from larger matrix, starting at a pointer representing the quadrant.
// src_size and sub_size refers to the size of the original resp. submatrix (stride). 

void get_submatrix(const int* src, int* dst, int src_row, int src_col, int src_size, int sub_size) {
    for (int i = 0; i < sub_size; i++) {
        const int* src_ptr = src + (src_row + i) * src_size + src_col;
        int* dst_ptr = dst + i * sub_size;
        memcpy(dst_ptr, src_ptr, sub_size * sizeof(int));
    }
}

// Add submatrix result back to destination matrix
// src_size and sub_size refers to the size of the original resp. submatrix (stride). 

void add_to_submatrix(int* dst, const int* src, int dst_row, int dst_col, int dst_size, int sub_size) {
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            dst[(dst_row + i) * dst_size + (dst_col + j)] += src[i * sub_size + j];
        }
    }
}
//Because we copy submatrices into contiguous memory we do not need to consider the stride here
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

// Strassen multiplication. Max_depth is the recursion level
void strassen(int* A, int* B, int* C, int size, int depth, int max_depth) {
    if (size == 2 || depth >= max_depth) {
        standard_matrix_multiplication(size, A, B, C);
        return;
    }

    int half = size / 2;
    int block = half * half;

    int* temp = calloc(25 * block, sizeof(int));

    // Allocate temporary submatrices from buffer
    int* A11 = temp;
    int* A12 = temp + block;
    int* A21 = temp + 2 * block;
    int* A22 = temp + 3 * block;

    int* B11 = temp + 4 * block;
    int* B12 = temp + 5 * block;
    int* B21 = temp + 6 * block;
    int* B22 = temp + 7 * block;

    int* M1 = temp + 8 * block;
    int* M2 = temp + 9 * block;
    int* M3 = temp + 10 * block;
    int* M4 = temp + 11 * block;
    int* M5 = temp + 12 * block;
    int* M6 = temp + 13 * block;
    int* M7 = temp + 14 * block;
    
    int* T1 = temp + 15 * block;
    int* T2 = temp + 16 * block;
    int* T3 = temp + 17 * block;
    int* T4 = temp + 18 * block;
    int* T5 = temp + 19 * block;
    int* T6 = temp + 20 * block;
    int* T7 = temp + 21 * block;
    int* T8 = temp + 22 * block;
    int* T9 = temp + 23 * block;
    int* T10 = temp + 24 * block;

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

    // M1 = (A11 + A22) * (B11 + B22)
    #pragma omp task
    {add_matrix(A11, A22, T1, half);
    add_matrix(B11, B22, T2, half);
    strassen(T1, T2, M1, half, depth + 1, max_depth);}

    // M2 = (A21 + A22) * B11
    #pragma omp task
    {add_matrix(A21, A22, T3, half);
    strassen(T3, B11, M2, half, depth + 1, max_depth);}

    // M3 = A11 * (B12 - B22)
    #pragma omp task
    {sub_matrix(B12, B22, T4, half);
    strassen(A11, T4, M3, half, depth + 1, max_depth);}

    // M4 = A22 * (B21 - B11)
    #pragma omp task
    {sub_matrix(B21, B11, T5, half);
    strassen(A22, T5, M4, half, depth + 1, max_depth);}

    // M5 = (A11 + A12) * B22
    #pragma omp task
    {add_matrix(A11, A12, T6, half);
    strassen(T6, B22, M5, half, depth + 1, max_depth);}

    // M6 = (A21 - A11) * (B11 + B12)
    #pragma omp task
    {sub_matrix(A21, A11, T7, half);
    add_matrix(B11, B12, T8, half);
    strassen(T7, T8, M6, half, depth + 1, max_depth);}

    // M7 = (A12 - A22) * (B21 + B22)
    #pragma omp task  
    {sub_matrix(A12, A22, T9, half);
    add_matrix(B21, B22, T10, half);
    strassen(T9, T10, M7, half, depth + 1, max_depth);}

    #pragma omp taskwait

    #pragma omp task
    {for (int i = 0; i < block; i++)
        T1[i] = M1[i] + M4[i] - M5[i] + M7[i];
    add_to_submatrix(C, T1, 0, 0, size, half);}

    #pragma omp task
    {for (int i = 0; i < block; i++)
        T2[i] = M3[i] + M5[i];
    add_to_submatrix(C, T2, 0, half, size, half);}

    #pragma omp task
    {for (int i = 0; i < block; i++)
        T3[i] = M2[i] + M4[i];
    add_to_submatrix(C, T3, half, 0, size, half);}

    #pragma omp task
    {for (int i = 0; i < block; i++)
        T4[i] = M1[i] - M2[i] + M3[i] + M6[i];
    add_to_submatrix(C, T4, half, half, size, half);}

    #pragma omp taskwait


    free(temp);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Input should be: ./a.out [matrix size] [recursion_depth] [num_threads] [0 or 1, check if result is correct with standard mult.]\n");
        return -1;
    }

    int N = atoi(argv[1]);
    int max_depth = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    int check_flag = atoi(argv[4]);

    int size = N * N;
    int* A = calloc(size, sizeof(int));
    int* B = calloc(size, sizeof(int));
    int* C = calloc(size, sizeof(int));
    int* C_check = calloc(size, sizeof(int));

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

    double start_time = get_wall_seconds(); //Start timing

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        {strassen(A, B, C, N, 0, max_depth);}
    }
    
    double end_time = get_wall_seconds();
    printf("Strassen multiplication time: %f seconds\n", end_time - start_time);

    // Below checks if the result is correct
    if (check_flag == 1) {
    
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
    if (correct == 0) {
        printf("Strassen multiplication result is incorrect.\n");
    } else {
        printf("Strassen multiplication result is correct.\n");
    }
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
    return 0;
}