#include <assert.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

// C definition of dgemm
extern "C" {
void dgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);
}

template <typename T>
void profile_gemm(int m, int n, int k)
{
    T* A = (T*)aligned_alloc(64, m * k * sizeof(T));
    T* B = (T*)aligned_alloc(64, k * n * sizeof(T));
    T* C = (T*)aligned_alloc(64, m * n * sizeof(T));

    // Fill A and B with random values
    for (int i = 0; i < m * k; i++) {
        A[i] = (T)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (T)rand() / RAND_MAX;
    }

    int n_timing = 20;
    double timing_min = std::numeric_limits<double>::max();

    char transa = 'N';
    char transb = 'N';
    T alpha = 1.0;
    T beta = 0.0;
    for (int i = 0; i < n_timing; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C,
               &m);
        auto end = std::chrono::high_resolution_clock::now();
        double timing = std::chrono::duration<double>(end - start).count();

        if (timing < timing_min) {
            timing_min = timing;
        }
    }

    double gflops = (2.0 * m * n * k) / timing_min / 1e9;

    std::cout << "m = " << m << ", n = " << n << ", k = " << k
              << ", timing = " << timing_min << " s, gflops = " << gflops
              << std::endl;

    free(A);
    free(B);
    free(C);
}

int main()
{
    // Do a 2000x2000 matmul as a warmup
    {
        double* A = (double*)aligned_alloc(64, 2000 * 2000 * sizeof(double));
        double* B = (double*)aligned_alloc(64, 2000 * 2000 * sizeof(double));
        double* C = (double*)aligned_alloc(64, 2000 * 2000 * sizeof(double));

        for (int i = 0; i < 2000 * 2000; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
            C[i] = 0;
        }

        for (int j = 0; j < 2000; j++) {
            for (int p = 0; p < 2000; p++) {
                for (int i = 0; i < 2000; i++) {
                    C[i + j * 2000] += A[i + p * 2000] * B[p + j * 2000];
                }
            }
        }

        free(A);
        free(B);
        free(C);
    }

    std::cout << "=================================" << std::endl;
    std::cout << "profiling dgemm" << std::endl;
    std::cout << "=================================" << std::endl;

    for (int n = 420; n <= 4000; n += 420) {
        profile_gemm<double>(n, n, 180);
    }

    return 0;
}
