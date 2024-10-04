#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <chrono>

#include "optimized/rotc.h"

template <typename T>
void profile_rotc(
    int m,
    int n,
    int k,
    void (*rotc)(char side, char dir, bool startup, bool shutdown, int m, int n, int k, T *A, int lda, const T *C, int ldc, const T *S, int lds))
{

    T *A = (T *)aligned_alloc(64, m * n * sizeof(T));
    T *C = (T *)aligned_alloc(64, k * n * sizeof(T));
    T *S = (T *)aligned_alloc(64, k * n * sizeof(T));

    // Fill A with random values
    for (int i = 0; i < m * n; i++)
    {
        A[i] = (T)rand() / RAND_MAX;
    }

    // Fill C and S with random rotation values
    for (int i = 0; i < k * n; i++)
    {
        T angle = (T)rand() / RAND_MAX * 2 * M_PI;
        C[i] = cos(angle);
        S[i] = sin(angle);
    }

    T *A_copy = (T *)aligned_alloc(64, m * n * sizeof(T));
    std::copy(A, A + m * n, A_copy);

    int n_timing = 20;
    int n_repeat = 5;
    double timing_min = std::numeric_limits<double>::max();

    for (int i = 0; i < n_timing; i++)
    {
        std::copy(A_copy, A_copy + m * n, A);
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < n_repeat; j++)
        {
            rotc('R', 'F', false, false, m, n, k, A, m, C, n, S, n);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timing = std::chrono::duration<double>(end - start).count() / n_repeat;

        if (timing < timing_min)
        {
            timing_min = timing;
        }
    }

    double gflops = (6.0 * m * k * (n-k)) / timing_min / 1e9;

    std::cout << "m = " << m << ", n = " << n << ", k = " << k << ", timing = " << timing_min << " s, gflops = " << gflops << std::endl;

    free(A);
    free(C);
    free(S);
    free(A_copy);
}

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "profiling drotc" << std::endl;
    std::cout << "=================================" << std::endl;

    profile_rotc<double>(1000, 1000, 180, drotc);

    return 0;
}
