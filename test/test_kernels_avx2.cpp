#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>

#include "optimized/kernels_avx2.h"

template <typename T>
void reference_kernel(
    int mr,
    int kr,
    int n,
    T *A,
    const T *P)
{
    for (int p = 0; p < kr; p++)
    {
        for (int j = 0; j < n; j++)
        {

            T c = P[2 * (j * kr + p)];
            T s = P[2 * (j * kr + p) + 1];
            int l = j + kr - 1 - p;
            for (int i = 0; i < mr; i++)
            {
                T temp = c * A[i + l * mr] + s * A[i + (l + 1) * mr];
                A[i + (l + 1) * mr] = -s * A[i + l * mr] + c * A[i + (l + 1) * mr];
                A[i + l * mr] = temp;
            }
        }
    }
}

template <typename T>
void test_kernel(
    int mr,
    int kr,
    void (*kernel)(int n, T *A, const T *P))
{
    for (int n = 1; n < 10; n++)
    {
        T *A = new T[mr * (n + kr)];
        T *P = new T[2 * kr * n];

        // Fill A with random values
        for (int i = 0; i < mr * (n + kr); i++)
        {
            A[i] = (T)rand() / RAND_MAX;
        }
        // Fill P with random rotation values
        for (int i = 0; i < 2 * kr * n; i += 2)
        {
            T angle = (T)rand() / RAND_MAX * 2 * M_PI;
            P[i] = cos(angle);
            P[i + 1] = sin(angle);
        }

        // Calculate reference result using reference_kernel
        T *A_ref = new T[mr * (n + kr)];
        std::copy(A, A + mr * (n + kr), A_ref);
        reference_kernel(mr, kr, n, A_ref, P);

        // Calculate result using kernel
        kernel(n, A, P);

        // Compare results
        // Note, we only require equality up to a tolerance, but we
        // can actually use exact equality if desired.
        const T tol = 1.0e10 * std::numeric_limits<T>::epsilon();
        T err = 0;
        for (int i = 0; i < mr * (n + kr); i++)
        {
            err = std::max(err, std::abs(A[i] - A_ref[i]));
        }

        if (err < tol)
        {
            std::cout << "Test passed, mr = " << mr << ", kr = " << kr << ", n = " << n << std::endl;
        }
        else
        {
            std::cout << "Test failed, mr = " << mr << ", kr = " << kr << ", n = " << n << std::endl;
            std::cout << "Error: " << err << std::endl;
            for (int i = 0; i < mr * (n + kr); i++)
            {
                if (std::abs(A[i] - A_ref[i]) > tol)
                {
                    std::cout << "A[" << i << "] = " << A[i] << ", A_ref[" << i << "] = " << A_ref[i] << std::endl;
                }
            }
        }

        delete[] A_ref;
        delete[] A;
        delete[] P;
    }
}

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "Testing avx 2 kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_kernel<double>(12, 3, drotc_kernel_12xnx3);
    test_kernel<double>(12, 1, drotc_kernel_12xnx1);

    test_kernel<float>(24, 3, srotc_kernel_24xnx3);
    test_kernel<float>(24, 1, srotc_kernel_24xnx1);

    return 0;
}
