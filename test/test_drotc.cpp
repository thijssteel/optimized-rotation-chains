#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "optimized/drotc.h"

template <typename T>
void drotc_reference(
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
    // TODO: add startup, shutdown, left, backward

    for (int p = 0; p < k; p++)
    {
        for (int j = k - 1 - p; j < n - p; j++)
        {
            T c = C[j + p * ldc];
            T s = S[j + p * lds];
            for (int i = 0; i < m; i++)
            {
                T temp = c * A[i + j * lda] + s * A[i + (j + 1) * lda];
                A[i + (j + 1) * lda] = -s * A[i + j * lda] + c * A[i + (j + 1) * lda];
                A[i + j * lda] = temp;
            }
        }
    }
}

template <typename T>
void test_drotc(
    int m,
    int n,
    int k,
    void (*rotc)(char side, char dir, bool startup, bool shutdown, int m, int n, int k, T *A, int lda, const T *C, int ldc, const T *S, int lds))
{

    T *A = new T[m * (n+1)];

    T *C = new T[k * n];
    T *S = new T[k * n];

    // Fill A with random values
    for (int i = 0; i < m * (n+1); i++)
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

    // Calculate reference result using drotc_reference
    T *A_ref = new T[m * (n+1)];
    std::copy(A, A + m * (n+1), A_ref);
    drotc_reference(m, n, k, A_ref, m, C, n, S, n);

    // Calculate result using optimized drotc
    rotc('R', 'F', false, false, m, n, k, A, m, C, n, S, n);

    // Compare results
    // Note, we only require equality up to a tolerance, but we
    // can actually use exact equality if desired.
    const T tol = 1.0e10 * std::numeric_limits<T>::epsilon();
    T err = 0;
    for (int i = 0; i < m * (n+1); i++)
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
        // for (int i = 0; i < m * (n+1); i++)
        // {
        //     if (std::abs(A[i] - A_ref[i]) > tol)
        //     {
        //         std::cout << "A[" << i << "] = " << A[i] << ", A_ref[" << i << "] = " << A_ref[i] << std::endl;
        //     }
        // }
    }

    delete[] A_ref;
    delete[] A;
    delete[] C;
    delete[] S;
}

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "Testing drotc" << std::endl;
    std::cout << "=================================" << std::endl;

    // test_drotc<double>(1000, 300, 100, drotc);

    for(int m = 100; m <= 1000; m += 100)
    {
        for(int n = 100; n <= 1000; n += 100)
        {
            for(int k = 60; k <= 300; k += 60)
            {
                // test_drotc<float>(m, n, k, srotc);
                test_drotc<double>(m, n, k, drotc);
            }
        }
    }

    return 0;
}
