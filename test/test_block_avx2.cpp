#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "optimized/rotc.h"

template <typename T>
void reference_block(
    int m,
    int n,
    int k,
    T *A,
    int lda,
    const T *C,
    int ldc,
    const T *S,
    int lds)
{
    for (int p = 0; p < k; p++)
    {
        for (int j = 0; j < n; j++)
        {
            int g = j + k - 1 - p;
            T c = C[g + p * ldc];
            T s = S[g + p * lds];
            for (int i = 0; i < m; i++)
            {
                T temp = c * A[i + g * lda] + s * A[i + (g + 1) * lda];
                A[i + (g + 1) * lda] = -s * A[i + g * lda] + c * A[i + (g + 1) * lda];
                A[i + g * lda] = temp;
            }
        }
    }
}

template <typename T>
void test_block(
    int m,
    int n,
    int k,
    void (*rotation_block)(int m, int n, int k, double *A, int lda, double *Ap, const double *C, int ldc, const double *S, int lds, double *P))
{
    T *A = new T[m * (n + k)];

    T *C = new T[k * (n + k)];
    T *S = new T[k * (n + k)];

    // Fill A with random values
    for (int i = 0; i < m * (n + k); i++)
    {
        A[i] = (T)rand() / RAND_MAX;
    }

    // Fill C and S with random rotation values
    for (int i = 0; i < k * (n + k); i++)
    {
        T angle = (T)rand() / RAND_MAX * 2 * M_PI;
        C[i] = cos(angle);
        S[i] = sin(angle);
    }

    // Calculate reference result using reference_block
    T *A_ref = new T[m * (n + k)];
    std::copy(A, A + m * (n + k), A_ref);
    reference_block(m, n, k, A_ref, m, C, n + k, S, n + k);

    // Calculate result using block
    T *Ap = (double*)aligned_alloc(64, (m + 24) * (n + k) * sizeof(T));
    T *P = (double*)aligned_alloc(64, 2 * k * n * sizeof(T));
    rotation_block(m, n, k, A, m, Ap, C, n + k, S, n + k, P);

    // Compare results
    // Note, we only require equality up to a tolerance, but we
    // can actually use exact equality if desired.
    const T tol = 1.0e10 * std::numeric_limits<T>::epsilon();
    T err = 0;
    for (int i = 0; i < m * (n + k); i++)
    {
        err = std::max(err, std::abs(A[i] - A_ref[i]));
    }

    if (err < tol)
    {
        std::cout << "Test passed, m = " << m << ", n = " << n << ", k = " << k << std::endl;
    }
    else
    {
        std::cout << "Test failed, m = " << m << ", n = " << n << ", k = " << k << std::endl;
        std::cout << "Error: " << err << std::endl;
        for (int i = 0; i < m * (n + k); i++)
        {
            if (std::abs(A[i] - A_ref[i]) > tol)
            {
                std::cout << "A[" << i << "] = " << A[i] << ", A_ref[" << i << "] = " << A_ref[i] << std::endl;
            }
        }
    }

    delete[] A_ref;
    delete[] A;
    delete[] C;
    delete[] S;
    free(Ap);
    free(P);
}

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "Testing trapezoidal block" << std::endl;
    std::cout << "=================================" << std::endl;

    // test_block<double>(12, 12, 4, drotc_pipeline_block_right);

    for( int m = 24; m <= 28; m ++ )
    {
        for( int n = 12; n <= 24; n++ )
        {
            for( int k = 1; k <= 6; k ++ )
            {
                test_block<double>(m, n, k, drotc_pipeline_block_right);
            }
        }
    } 

    return 0;
}
