#include <assert.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#include "optimized/drotc.h"
#include "optimized/drotc_params.h"

template <typename T>
void profile_rotc(int m,
                  int n,
                  int k,
                  void (*rotc)(char side,
                               char dir,
                               bool startup,
                               bool shutdown,
                               bool trans,
                               int m,
                               int n,
                               int k,
                               T* A,
                               int lda,
                               const T* C,
                               int ldc,
                               const T* S,
                               int lds))
{
    T* A = (T*)aligned_alloc(64, m * (n + 1) * sizeof(T));
    T* C = (T*)aligned_alloc(64, k * n * sizeof(T));
    T* S = (T*)aligned_alloc(64, k * n * sizeof(T));

    // Fill A with random values
    for (int i = 0; i < m * (n + 1); i++) {
        A[i] = (T)rand() / RAND_MAX;
    }

    // Fill C and S with random rotation values
    for (int i = 0; i < k * n; i++) {
        T angle = (T)rand() / RAND_MAX * 2 * M_PI;
        C[i] = cos(angle);
        S[i] = sin(angle);
    }

    T* A_copy = (T*)aligned_alloc(64, m * (n + 1) * sizeof(T));
    std::copy(A, A + m * (n + 1), A_copy);

    int n_timing = 20;
    int n_repeat = 5;
    double timing_min = std::numeric_limits<double>::max();

    for (int i = 0; i < n_timing; i++) {
        std::copy(A_copy, A_copy + m * (n + 1), A);
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < n_repeat; j++) {
            rotc('R', 'F', false, false, m, n, k, A, m, C, n, S, n);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timing =
            std::chrono::duration<double>(end - start).count() / n_repeat;

        if (timing < timing_min) {
            timing_min = timing;
        }
    }

    double gflops = (6.0 * m * k * (n - k + 1)) / timing_min / 1e9;

    std::cout << "m = " << m << ", n = " << n << ", k = " << k
              << ", timing = " << timing_min << " s, gflops = " << gflops
              << std::endl;

    free(A);
    free(C);
    free(S);
    free(A_copy);
}

template <typename T>
void profile_rotc_prepack(int m,
                          int n,
                          int k,
                          void (*rotc_prepacked)(char side,
                                                 char dir,
                                                 bool startup,
                                                 bool shutdown,
                                                 bool trans,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 T* Ap,
                                                 int lda,
                                                 const T* C,
                                                 int ldc,
                                                 const T* S,
                                                 int lds))
{
    int mp = (m + MR - 1) / MR * MR;

    T* Ap = (T*)aligned_alloc(64, mp * (n + 1) * sizeof(T));
    T* C = (T*)aligned_alloc(64, k * n * sizeof(T));
    T* S = (T*)aligned_alloc(64, k * n * sizeof(T));

    // Fill A with random values
    for (int i = 0; i < mp * (n + 1); i++) {
        Ap[i] = (T)rand() / RAND_MAX;
    }

    // Fill C and S with random rotation values
    for (int i = 0; i < k * n; i++) {
        T angle = (T)rand() / RAND_MAX * 2 * M_PI;
        C[i] = cos(angle);
        S[i] = sin(angle);
    }

    T* Ap_copy = (T*)aligned_alloc(64, mp * (n + 1) * sizeof(T));
    std::copy(Ap, Ap + mp * (n + 1), Ap_copy);

    int n_timing = 20;
    int n_repeat = 5;
    double timing_min = std::numeric_limits<double>::max();

    for (int i = 0; i < n_timing; i++) {
        std::copy(Ap_copy, Ap_copy + mp * (n + 1), Ap);
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < n_repeat; j++) {
            rotc_prepacked('R', 'F', false, false, mp, n, k, Ap, n + 1, C, n, S,
                           n);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timing =
            std::chrono::duration<double>(end - start).count() / n_repeat;

        if (timing < timing_min) {
            timing_min = timing;
        }
    }

    double gflops = (6.0 * m * k * (n - k + 1)) / timing_min / 1e9;

    std::cout << "m = " << m << ", n = " << n << ", k = " << k
              << ", timing = " << timing_min << " s, gflops = " << gflops
              << std::endl;

    free(Ap);
    free(C);
    free(S);
    free(Ap_copy);
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
    std::cout << "profiling drotc" << std::endl;
    std::cout << "=================================" << std::endl;

    for (int n = 420; n <= 4000; n += 420) {
        profile_rotc<double>(n, n, 180, drotc);
    }

    std::cout << "=================================" << std::endl;
    std::cout << "profiling prepacked drotc" << std::endl;
    std::cout << "=================================" << std::endl;

    for (int n = 420; n <= 4000; n += 420) {
        profile_rotc_prepack<double>(n, n, 180, drotc_prepacked);
    }

    return 0;
}
