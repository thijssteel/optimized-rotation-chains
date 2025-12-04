#include <assert.h>

#include <cmath>
#include <cstring>
#include <iostream>

#include "optimized/drotc_kernels.h"
#include "optimized/drotc_params.h"

void reference_kernel(int mr,
                      int kr,
                      int n,
                      double* A,
                      const double* C,
                      int ldc,
                      const double* S,
                      int lds)
{
    for (int p = 0; p < kr; p++) {
        for (int j = 0; j < n; j++) {
            double c = C[j + p * ldc];
            double s = S[j + p * lds];
            // std::cout << "c = " << c << " s = " << s << std::endl;
            int g = j + kr - 1 - p;
            for (int i = 0; i < mr; i++) {
                // std::cout << "A[" << i << "] = " << A[i + g * mr] << " , " <<
                // A[i + (g + 1) * mr] << std::endl;
                double temp = c * A[i + g * mr] + s * A[i + (g + 1) * mr];
                A[i + (g + 1) * mr] =
                    -s * A[i + g * mr] + c * A[i + (g + 1) * mr];
                A[i + g * mr] = temp;
            }
        }
    }
}

void reference_kernel_backwards(int mr,
                                int kr,
                                int n,
                                double* A,
                                const double* C,
                                int ldc,
                                const double* S,
                                int lds)
{
    for (int p = 0; p < kr; p++) {
        for (int j = n - 1; j >= 0; j--) {
            double c = C[j + p * ldc];
            double s = S[j + p * lds];
            // std::cout << "c = " << c << " s = " << s << std::endl;
            int g = j + p;
            for (int i = 0; i < mr; i++) {
                // std::cout << "A[" << i << "] = " << A[i + g * mr] << " , " <<
                // A[i + (g + 1) * mr] << std::endl;
                double temp = c * A[i + g * mr] + s * A[i + (g + 1) * mr];
                A[i + (g + 1) * mr] =
                    -s * A[i + g * mr] + c * A[i + (g + 1) * mr];
                A[i + g * mr] = temp;
            }
        }
    }
}

void reference_packing(int m, int n, double* A, int lda, double* Ap)
{
    int ib = 0;
    for (; ib + (MR - 1) < m; ib += MR) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < MR; i++) {
                Ap[ib * n + j * MR + i] = A[i + ib + j * lda];
            }
        }
    }
    if (ib < m) {
        // Less than mr rows remaining
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m - ib; i++) {
                Ap[ib * n + j * MR + i] = A[i + ib + j * lda];
            }
        }
    }
}

template <typename T>
void test_kernel_mrxnxkr(
    bool backward,
    void (*kernel)(int n, T* A, const T* C, int ldc, const T* S, int lds))
{
    for (int n = 1; n <= 10; n++) {
        T* A = new T[MR * (n + KR)];
        T* C = new T[n * KR];
        T* S = new T[n * KR];

        // Fill A with random values
        for (int i = 0; i < MR * (n + KR); i++) {
            A[i] = (T)rand() / RAND_MAX;
        }
        // Fill C and S with random rotation values
        for (int i = 0; i < KR * n; i++) {
            T angle = (T)rand() / RAND_MAX * 2 * M_PI;
            C[i] = cos(angle);
            S[i] = sin(angle);
        }

        // Calculate reference result using reference_kernel
        T* A_ref = new T[MR * (n + KR)];
        std::copy(A, A + MR * (n + KR), A_ref);
        if (backward) {
            reference_kernel_backwards(MR, KR, n, A_ref, C, n, S, n);
        }
        else {
            reference_kernel(MR, KR, n, A_ref, C, n, S, n);
        }

        // Calculate result using kernel
        kernel(n, A, C, n, S, n);

        // Compare results
        // Note, we only require equality up to a tolerance, but we
        // can actually use exact equality if desired.
        const T tol = 10.0e1 * std::numeric_limits<T>::epsilon();
        T err = 0;
        for (int i = 0; i < MR * (n + KR); i++) {
            err = std::max(err, std::abs(A[i] - A_ref[i]));
        }

        if (err < tol) {
            std::cout << "Test passed, MR = " << MR << ", KR = " << KR
                      << ", n = " << n << std::endl;
        }
        else {
            std::cout << "Test failed, MR = " << MR << ", KR = " << KR
                      << ", n = " << n << std::endl;
            std::cout << "Error: " << err << std::endl;
            for (int i = 0; i < MR * (n + KR); i++) {
                // if (std::abs(A[i] - A_ref[i]) > tol)
                {
                    std::cout << "A[" << i << "] = " << A[i] << ", A_ref[" << i
                              << "] = " << A_ref[i] << std::endl;
                }
            }
        }

        delete[] A_ref;
        delete[] A;
        delete[] C;
        delete[] S;
    }
}

template <typename T>
void test_kernel_mrxnx1(bool backward,
                        void (*kernel)(int n, T* A, const T* C, const T* S))
{
    for (int n = 1; n <= 10; n++) {
        T* A = new T[MR * (n + 1)];
        T* C = new T[n];
        T* S = new T[n];

        // Fill A with random values
        for (int i = 0; i < MR * (n + 1); i++) {
            A[i] = (T)rand() / RAND_MAX;
        }
        // Fill C and S with random rotation values
        for (int i = 0; i < n; i++) {
            T angle = (T)rand() / RAND_MAX * 2 * M_PI;
            C[i] = cos(angle);
            S[i] = sin(angle);
        }

        // Calculate reference result using reference_kernel
        T* A_ref = new T[MR * (n + 1)];
        std::copy(A, A + MR * (n + 1), A_ref);
        if (backward) {
            reference_kernel_backwards(MR, 1, n, A_ref, C, n, S, n);
        }
        else {
            reference_kernel(MR, 1, n, A_ref, C, n, S, n);
        }

        // Calculate result using kernel
        kernel(n, A, C, S);

        // Compare results
        // Note, we only require equality up to a tolerance, but we
        // can actually use exact equality if desired.
        const T tol = 10.0e1 * std::numeric_limits<T>::epsilon();
        T err = 0;
        for (int i = 0; i < MR * (n + 1); i++) {
            err = std::max(err, std::abs(A[i] - A_ref[i]));
        }

        if (err < tol) {
            std::cout << "Test passed, MR = " << MR << ", KR = " << KR
                      << ", n = " << n << std::endl;
        }
        else {
            std::cout << "Test failed, MR = " << MR << ", KR = " << KR
                      << ", n = " << n << std::endl;
            std::cout << "Error: " << err << std::endl;
            for (int i = 0; i < MR * (n + 1); i++) {
                // if (std::abs(A[i] - A_ref[i]) > tol)
                {
                    std::cout << "A[" << i << "] = " << A[i] << ", A_ref[" << i
                              << "] = " << A_ref[i] << std::endl;
                }
            }
        }

        delete[] A_ref;
        delete[] A;
        delete[] C;
        delete[] S;
    }
}

void test_packing(
    void (*pack_kernel)(int m, int n, const double* A, int lda, double* Ap),
    void (*unpack_kernel)(int m, int n, const double* Ap, double* A, int lda))
{
    for (int m = 1; m < 28; m += 3) {
        for (int n = 1; n < 10; n += 3) {
            int mp = (m + MR - 1) / MR * MR;
            double* A = new double[m * n];
            double* A_copy = new double[m * n];
            double* Ap = new double[mp * n];
            double* Ap_ref = new double[mp * n];

            // Fill A with random values
            for (int i = 0; i < m * n; i++) {
                A[i] = (double)rand() / RAND_MAX;
            }

            std::copy(A, A + m * n, A_copy);

            // Fill Ap and Ap_ref with zeros
            for (int i = 0; i < mp * n; i++) {
                Ap[i] = 0;
                Ap_ref[i] = 0;
            }

            // Calculate reference result using reference_packing
            pack_kernel(m, n, A, m, Ap_ref);

            // Calculate result using kernel
            pack_kernel(m, n, A, m, Ap);

            // Unpack Ap into A
            unpack_kernel(m, n, Ap, A, m);

            // Compare results
            double err = 0;
            for (int i = 0; i < mp * n; i++) {
                err = std::max(err, std::abs(Ap[i] - Ap_ref[i]));
            }

            double err2 = 0;
            for (int i = 0; i < m * n; i++) {
                err2 = std::max(err2, std::abs(A[i] - A_copy[i]));
            }

            if (err == 0 and err2 == 0) {
                std::cout << "Test passed, m = " << m << ", n = " << n
                          << std::endl;
            }
            else {
                std::cout << "Test failed, m = " << m << ", n = " << n
                          << std::endl;
                std::cout << "Packing error: " << err << std::endl;
                std::cout << "Unpacking error: " << err2 << std::endl;
            }

            delete[] A;
            delete[] A_copy;
            delete[] Ap;
            delete[] Ap_ref;
        }
    }
}

int main()
{
    std::cout << "=================================" << std::endl;
    std::cout << "Testing forward full kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_kernel_mrxnxkr(false, drotc_kernel_mrxnxkr);


    std::cout << "=================================" << std::endl;
    std::cout << "Testing forward edge kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_kernel_mrxnx1(false, drotc_kernel_mrxnx1);

    std::cout << "=================================" << std::endl;
    std::cout << "Testing backward full kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_kernel_mrxnxkr(true, drotc_kernel_mrxnxkr_backward);

    std::cout << "=================================" << std::endl;
    std::cout << "Testing backward edge kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_kernel_mrxnx1(true, drotc_kernel_mrxnx1_backward);

    std::cout << "=================================" << std::endl;
    std::cout << "Testing packing kernels" << std::endl;
    std::cout << "=================================" << std::endl;

    test_packing(drotc_pack_A, drotc_unpack_A);

    return 0;
}
