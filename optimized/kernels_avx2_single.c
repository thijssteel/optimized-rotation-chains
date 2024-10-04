#include "immintrin.h"

#include "./kernels_avx2.h"

void srotc_kernel_24xnx3(int n,
                         float *restrict A,
                         const float *restrict P)
{
    // Number of rows handled by the kernel
    const int mr = 24;
    // Size of the wavefront
    const int kr = 3;
    // Number of values in a vector register
    const int nv = 8;

    // Load initial values of A
    __m256 a_0_0 = _mm256_loadu_ps(A);
    __m256 a_1_0 = _mm256_loadu_ps(&A[nv]);
    __m256 a_2_0 = _mm256_loadu_ps(&A[2 * nv]);

    __m256 a_0_1 = _mm256_loadu_ps(&A[mr]);
    __m256 a_1_1 = _mm256_loadu_ps(&A[mr + nv]);
    __m256 a_2_1 = _mm256_loadu_ps(&A[mr + 2 * nv]);

    __m256 a_0_2 = _mm256_loadu_ps(&A[2 * mr]);
    __m256 a_1_2 = _mm256_loadu_ps(&A[2 * mr + nv]);
    __m256 a_2_2 = _mm256_loadu_ps(&A[2 * mr + 2 * nv]);

    for (int i = 0; i < n; i++)
    {
        __m256 temp, c_vec, s_vec;

        // Load new values of A
        __m256 a_0_next = _mm256_loadu_ps(&A[3 * mr]);
        __m256 a_1_next = _mm256_loadu_ps(&A[3 * mr + nv]);
        __m256 a_2_next = _mm256_loadu_ps(&A[3 * mr + 2 * nv]);

        // Apply rotation 0 to values 2 and 3 (next)
        c_vec = _mm256_set1_ps(P[0]);
        s_vec = _mm256_set1_ps(P[1]);

        temp = _mm256_fmadd_ps(c_vec, a_0_2, _mm256_mul_ps(s_vec, a_0_next));
        a_0_2 = _mm256_fnmadd_ps(s_vec, a_0_2, _mm256_mul_ps(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_1_2, _mm256_mul_ps(s_vec, a_1_next));
        a_1_2 = _mm256_fnmadd_ps(s_vec, a_1_2, _mm256_mul_ps(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_2_2, _mm256_mul_ps(s_vec, a_2_next));
        a_2_2 = _mm256_fnmadd_ps(s_vec, a_2_2, _mm256_mul_ps(c_vec, a_2_next));
        a_2_next = temp;

        // Apply rotation 1 to values 1 and 2 (next)
        c_vec = _mm256_set1_ps(P[2 + 0]);
        s_vec = _mm256_set1_ps(P[2 + 1]);

        temp = _mm256_fmadd_ps(c_vec, a_0_1, _mm256_mul_ps(s_vec, a_0_next));
        a_0_1 = _mm256_fnmadd_ps(s_vec, a_0_1, _mm256_mul_ps(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_1_1, _mm256_mul_ps(s_vec, a_1_next));
        a_1_1 = _mm256_fnmadd_ps(s_vec, a_1_1, _mm256_mul_ps(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_2_1, _mm256_mul_ps(s_vec, a_2_next));
        a_2_1 = _mm256_fnmadd_ps(s_vec, a_2_1, _mm256_mul_ps(c_vec, a_2_next));
        a_2_next = temp;

        // Apply rotation 2 to values 0 and 1 (next)
        c_vec = _mm256_set1_ps(P[2 * 2 + 0]);
        s_vec = _mm256_set1_ps(P[2 * 2 + 1]);

        temp = _mm256_fmadd_ps(c_vec, a_0_0, _mm256_mul_ps(s_vec, a_0_next));
        a_0_0 = _mm256_fnmadd_ps(s_vec, a_0_0, _mm256_mul_ps(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_1_0, _mm256_mul_ps(s_vec, a_1_next));
        a_1_0 = _mm256_fnmadd_ps(s_vec, a_1_0, _mm256_mul_ps(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_ps(c_vec, a_2_0, _mm256_mul_ps(s_vec, a_2_next));
        a_2_0 = _mm256_fnmadd_ps(s_vec, a_2_0, _mm256_mul_ps(c_vec, a_2_next));
        a_2_next = temp;

        // Store value 0 (stored in next)
        _mm256_storeu_ps(&A[0], a_0_next);
        _mm256_storeu_ps(&A[nv], a_1_next);
        _mm256_storeu_ps(&A[2 * nv], a_2_next);

        // Increment pointers
        A += mr;
        P += 2 * kr;
    }

    // Store final values of A
    _mm256_storeu_ps(&A[0], a_0_0);
    _mm256_storeu_ps(&A[nv], a_1_0);
    _mm256_storeu_ps(&A[2 * nv], a_2_0);

    _mm256_storeu_ps(&A[mr + 0], a_0_1);
    _mm256_storeu_ps(&A[mr + nv], a_1_1);
    _mm256_storeu_ps(&A[mr + 2 * nv], a_2_1);

    _mm256_storeu_ps(&A[2 * mr + 0], a_0_2);
    _mm256_storeu_ps(&A[2 * mr + nv], a_1_2);
    _mm256_storeu_ps(&A[2 * mr + 2 * nv], a_2_2);
}

void srotc_kernel_24xnx1(int n,
                         float *restrict A,
                         const float *P)
{
    // Number of rows handled by the kernel
    const int mr = 24;
    // Size of the wavefront
    const int kr = 1;
    // Number of values in a vector register
    const int nv = 8;

    // Load initial values of A
    __m256 a_0_0 = _mm256_loadu_ps(A);
    __m256 a_1_0 = _mm256_loadu_ps(&A[nv]);
    __m256 a_2_0 = _mm256_loadu_ps(&A[2 * nv]);

    for (int i = 0; i < n; i++)
    {
        __m256 temp1, temp2, temp3, c_vec, s_vec;

        // Load new values of A
        __m256 a_0_next = _mm256_loadu_ps(&A[mr]);
        __m256 a_1_next = _mm256_loadu_ps(&A[mr + nv]);
        __m256 a_2_next = _mm256_loadu_ps(&A[mr + 2 * nv]);

        // Apply rotation 0 to values 0 and 1 (next)
        c_vec = _mm256_set1_ps(P[0]);
        s_vec = _mm256_set1_ps(P[1]);

        temp1 = _mm256_fmadd_ps(c_vec, a_0_0, _mm256_mul_ps(s_vec, a_0_next));
        a_0_0 = _mm256_fnmadd_ps(s_vec, a_0_0, _mm256_mul_ps(c_vec, a_0_next));
        a_0_next = temp1;

        temp2 = _mm256_fmadd_ps(c_vec, a_1_0, _mm256_mul_ps(s_vec, a_1_next));
        a_1_0 = _mm256_fnmadd_ps(s_vec, a_1_0, _mm256_mul_ps(c_vec, a_1_next));
        a_1_next = temp2;

        temp3 = _mm256_fmadd_ps(c_vec, a_2_0, _mm256_mul_ps(s_vec, a_2_next));
        a_2_0 = _mm256_fnmadd_ps(s_vec, a_2_0, _mm256_mul_ps(c_vec, a_2_next));
        a_2_next = temp3;

        // Store value 0 (stored in next)
        _mm256_storeu_ps(&A[0], a_0_next);
        _mm256_storeu_ps(&A[nv], a_1_next);
        _mm256_storeu_ps(&A[2 * nv], a_2_next);

        // Increment pointers
        A += mr;
        P += 2 * kr;
    }

    // Store final values of A
    _mm256_storeu_ps(&A[0], a_0_0);
    _mm256_storeu_ps(&A[nv], a_1_0);
    _mm256_storeu_ps(&A[2 * nv], a_2_0);
}