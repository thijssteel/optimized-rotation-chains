#include "immintrin.h"
#include "stdio.h"

#include "./kernels_avx2.h"



void drotc_kernel_12xnx3(int n,
                         double *restrict A,
                         const double *restrict P)
{
    // Number of rows handled by the kernel
    const int mr = 12;
    // Size of the wavefront
    const int kr = 3;
    // Number of values in a vector register
    const int nv = 4;

    // Load initial values of A
    __m256d a_0_0 = _mm256_loadu_pd(A);
    __m256d a_1_0 = _mm256_loadu_pd(&A[nv]);
    __m256d a_2_0 = _mm256_loadu_pd(&A[2 * nv]);

    __m256d a_0_1 = _mm256_loadu_pd(&A[mr]);
    __m256d a_1_1 = _mm256_loadu_pd(&A[mr + nv]);
    __m256d a_2_1 = _mm256_loadu_pd(&A[mr + 2 * nv]);

    __m256d a_0_2 = _mm256_loadu_pd(&A[2 * mr]);
    __m256d a_1_2 = _mm256_loadu_pd(&A[2 * mr + nv]);
    __m256d a_2_2 = _mm256_loadu_pd(&A[2 * mr + 2 * nv]);

    for (int i = 0; i < n; i++)
    {
        __m256d temp, c_vec, s_vec;

        // Load new values of A
        __m256d a_0_next = _mm256_loadu_pd(&A[3 * mr]);
        __m256d a_1_next = _mm256_loadu_pd(&A[3 * mr + nv]);
        __m256d a_2_next = _mm256_loadu_pd(&A[3 * mr + 2 * nv]);

        // Apply rotation 0 to values 2 and 3 (next)
        c_vec = _mm256_set1_pd(P[0]);
        s_vec = _mm256_set1_pd(P[1]);

        temp = c_vec * a_0_2 + s_vec * a_0_next;
        a_0_2 = -s_vec * a_0_2 + c_vec * a_0_next;
        a_0_next = temp;

        temp = c_vec * a_1_2 + s_vec * a_1_next;
        a_1_2 = -s_vec * a_1_2 + c_vec * a_1_next;
        a_1_next = temp;

        temp = c_vec * a_2_2 + s_vec * a_2_next;
        a_2_2 = -s_vec * a_2_2 + c_vec * a_2_next;
        a_2_next = temp;

        // Apply rotation 1 to values 1 and 2 (next)
        c_vec = _mm256_set1_pd(P[2 + 0]);
        s_vec = _mm256_set1_pd(P[2 + 1]);

        temp = c_vec * a_0_1 + s_vec * a_0_next;
        a_0_1 = -s_vec * a_0_1 + c_vec * a_0_next;
        a_0_next = temp;

        temp = c_vec * a_1_1 + s_vec * a_1_next;
        a_1_1 = -s_vec * a_1_1 + c_vec * a_1_next;
        a_1_next = temp;

        temp = c_vec * a_2_1 + s_vec * a_2_next;
        a_2_1 = -s_vec * a_2_1 + c_vec * a_2_next;
        a_2_next = temp;

        // Apply rotation 2 to values 0 and 1 (next)
        c_vec = _mm256_set1_pd(P[2 * 2 + 0]);
        s_vec = _mm256_set1_pd(P[2 * 2 + 1]);

        temp = c_vec * a_0_0 + s_vec * a_0_next;
        a_0_0 = -s_vec * a_0_0 + c_vec * a_0_next;
        a_0_next = temp;

        temp = c_vec * a_1_0 + s_vec * a_1_next;
        a_1_0 = -s_vec * a_1_0 + c_vec * a_1_next;
        a_1_next = temp;

        temp = c_vec * a_2_0 + s_vec * a_2_next;
        a_2_0 = -s_vec * a_2_0 + c_vec * a_2_next;
        a_2_next = temp;

        // Store value 0 (stored in next)
        _mm256_storeu_pd(&A[0], a_0_next);
        _mm256_storeu_pd(&A[nv], a_1_next);
        _mm256_storeu_pd(&A[2 * nv], a_2_next);

        // Increment pointers
        A += mr;
        P += 2 * kr;
    }

    // Store final values of A
    _mm256_storeu_pd(&A[0], a_0_0);
    _mm256_storeu_pd(&A[nv], a_1_0);
    _mm256_storeu_pd(&A[2 * nv], a_2_0);

    _mm256_storeu_pd(&A[mr + 0], a_0_1);
    _mm256_storeu_pd(&A[mr + nv], a_1_1);
    _mm256_storeu_pd(&A[mr + 2 * nv], a_2_1);

    _mm256_storeu_pd(&A[2 * mr + 0], a_0_2);
    _mm256_storeu_pd(&A[2 * mr + nv], a_1_2);
    _mm256_storeu_pd(&A[2 * mr + 2 * nv], a_2_2);
}

void drotc_kernel_12xnx1(int n,
                         double *restrict A,
                         const double *P)
{
    // Number of rows handled by the kernel
    const int mr = 12;
    // Size of the wavefront
    const int kr = 1;
    // Number of values in a vector register
    const int nv = 4;

    // Load initial values of A
    __m256d a_0_0 = _mm256_loadu_pd(A);
    __m256d a_1_0 = _mm256_loadu_pd(&A[nv]);
    __m256d a_2_0 = _mm256_loadu_pd(&A[2 * nv]);

    for (int i = 0; i < n; i++)
    {
        __m256d temp1, temp2, temp3, c_vec, s_vec;

        // Load new values of A
        __m256d a_0_next = _mm256_loadu_pd(&A[mr]);
        __m256d a_1_next = _mm256_loadu_pd(&A[mr + nv]);
        __m256d a_2_next = _mm256_loadu_pd(&A[mr + 2 * nv]);

        // Apply rotation 0 to values 0 and 1 (next)
        c_vec = _mm256_set1_pd(P[0]);
        s_vec = _mm256_set1_pd(P[1]);

        temp1 = c_vec * a_0_0 + s_vec * a_0_next;
        a_0_0 = -s_vec * a_0_0 + c_vec * a_0_next;
        a_0_next = temp1;

        temp2 = c_vec * a_1_0 + s_vec * a_1_next;
        a_1_0 = -s_vec * a_1_0 + c_vec * a_1_next;
        a_1_next = temp2;

        temp3 = c_vec * a_2_0 + s_vec * a_2_next;
        a_2_0 = -s_vec * a_2_0 + c_vec * a_2_next;
        a_2_next = temp3;

        // Store value 0 (stored in next)
        _mm256_storeu_pd(&A[0], a_0_next);
        _mm256_storeu_pd(&A[nv], a_1_next);
        _mm256_storeu_pd(&A[2 * nv], a_2_next);

        // Increment pointers
        A += mr;
        P += 2 * kr;
    }

    // Store final values of A
    _mm256_storeu_pd(&A[0], a_0_0);
    _mm256_storeu_pd(&A[nv], a_1_0);
    _mm256_storeu_pd(&A[2 * nv], a_2_0);
}