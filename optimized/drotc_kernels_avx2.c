#include "drotc_params.h"
#include "drotc_kernels.h"

#include <immintrin.h>

void drotc_kernel_mrxnxkr(int n, double * A, const double * C, int ldc, const double * S, int lds)
{
    // Load initial values of A
    __m256d a_0_0 = _mm256_loadu_pd(A);
    __m256d a_1_0 = _mm256_loadu_pd(&A[4]);
    __m256d a_2_0 = _mm256_loadu_pd(&A[8]);

    __m256d a_0_1 = _mm256_loadu_pd(&A[MR]);
    __m256d a_1_1 = _mm256_loadu_pd(&A[MR + 4]);
    __m256d a_2_1 = _mm256_loadu_pd(&A[MR + 8]);

    __m256d a_0_2 = _mm256_loadu_pd(&A[2 * MR]);
    __m256d a_1_2 = _mm256_loadu_pd(&A[2 * MR + 4]);
    __m256d a_2_2 = _mm256_loadu_pd(&A[2 * MR + 8]);

    for (int i = 0; i < n; i++)
    {
        __m256d temp, c_vec, s_vec;

        // Load new values of A
        __m256d a_0_next = _mm256_loadu_pd(&A[3 * MR]);
        __m256d a_1_next = _mm256_loadu_pd(&A[3 * MR + 4]);
        __m256d a_2_next = _mm256_loadu_pd(&A[3 * MR + 8]);

        // Apply rotation 0 to values 2 and 3 (next)
        c_vec = _mm256_set1_pd(C[0]);
        s_vec = _mm256_set1_pd(S[0]);

        temp = _mm256_fmadd_pd(c_vec, a_0_2, _mm256_mul_pd(s_vec, a_0_next));
        a_0_2 = _mm256_fnmadd_pd(s_vec, a_0_2, _mm256_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_1_2, _mm256_mul_pd(s_vec, a_1_next));
        a_1_2 = _mm256_fnmadd_pd(s_vec, a_1_2, _mm256_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_2_2, _mm256_mul_pd(s_vec, a_2_next));
        a_2_2 = _mm256_fnmadd_pd(s_vec, a_2_2, _mm256_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        // Apply rotation 1 to values 1 and 2 (next)
        c_vec = _mm256_set1_pd(C[ldc]);
        s_vec = _mm256_set1_pd(S[lds]);

        temp = _mm256_fmadd_pd(c_vec, a_0_1, _mm256_mul_pd(s_vec, a_0_next));
        a_0_1 = _mm256_fnmadd_pd(s_vec, a_0_1, _mm256_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_1_1, _mm256_mul_pd(s_vec, a_1_next));
        a_1_1 = _mm256_fnmadd_pd(s_vec, a_1_1, _mm256_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_2_1, _mm256_mul_pd(s_vec, a_2_next));
        a_2_1 = _mm256_fnmadd_pd(s_vec, a_2_1, _mm256_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        // Apply rotation 2 to values 0 and 1 (next)
        c_vec = _mm256_set1_pd(C[2 * ldc]);
        s_vec = _mm256_set1_pd(S[2 * lds]);

        temp = _mm256_fmadd_pd(c_vec, a_0_0, _mm256_mul_pd(s_vec, a_0_next));
        a_0_0 = _mm256_fnmadd_pd(s_vec, a_0_0, _mm256_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_1_0, _mm256_mul_pd(s_vec, a_1_next));
        a_1_0 = _mm256_fnmadd_pd(s_vec, a_1_0, _mm256_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm256_fmadd_pd(c_vec, a_2_0, _mm256_mul_pd(s_vec, a_2_next));
        a_2_0 = _mm256_fnmadd_pd(s_vec, a_2_0, _mm256_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        // Store value 0 (stored in next)
        _mm256_storeu_pd(&A[0], a_0_next);
        _mm256_storeu_pd(&A[4], a_1_next);
        _mm256_storeu_pd(&A[8], a_2_next);

        // Increment pointers
        A += MR;
        C++;
        S++;
    }

    // Store final values of A
    _mm256_storeu_pd(&A[0], a_0_0);
    _mm256_storeu_pd(&A[4], a_1_0);
    _mm256_storeu_pd(&A[8], a_2_0);

    _mm256_storeu_pd(&A[MR + 0], a_0_1);
    _mm256_storeu_pd(&A[MR + 4], a_1_1);
    _mm256_storeu_pd(&A[MR + 8], a_2_1);

    _mm256_storeu_pd(&A[2 * MR + 0], a_0_2);
    _mm256_storeu_pd(&A[2 * MR + 4], a_1_2);
    _mm256_storeu_pd(&A[2 * MR + 8], a_2_2);
}

void drotc_kernel_mrxnx1(int n, double * A, const double * C, const double * S)
{
    // Load initial values of A
    __m256d a_0_0 = _mm256_loadu_pd(A);
    __m256d a_1_0 = _mm256_loadu_pd(&A[4]);
    __m256d a_2_0 = _mm256_loadu_pd(&A[8]);

    for (int i = 0; i < n; i++)
    {
        __m256d temp1, temp2, temp3, c_vec, s_vec;

        // Load new values of A
        __m256d a_0_next = _mm256_loadu_pd(&A[MR]);
        __m256d a_1_next = _mm256_loadu_pd(&A[MR + 4]);
        __m256d a_2_next = _mm256_loadu_pd(&A[MR + 8]);

        // Apply rotation 0 to values 0 and 1 (next)
        c_vec = _mm256_set1_pd(C[0]);
        s_vec = _mm256_set1_pd(S[0]);

        temp1 = _mm256_fmadd_pd(c_vec, a_0_0, _mm256_mul_pd(s_vec, a_0_next));
        a_0_0 = _mm256_fnmadd_pd(s_vec, a_0_0, _mm256_mul_pd(c_vec, a_0_next));
        a_0_next = temp1;

        temp2 = _mm256_fmadd_pd(c_vec, a_1_0, _mm256_mul_pd(s_vec, a_1_next));
        a_1_0 = _mm256_fnmadd_pd(s_vec, a_1_0, _mm256_mul_pd(c_vec, a_1_next));
        a_1_next = temp2;

        temp3 = _mm256_fmadd_pd(c_vec, a_2_0, _mm256_mul_pd(s_vec, a_2_next));
        a_2_0 = _mm256_fnmadd_pd(s_vec, a_2_0, _mm256_mul_pd(c_vec, a_2_next));
        a_2_next = temp3;

        // Store value 0 (stored in next)
        _mm256_storeu_pd(&A[0], a_0_next);
        _mm256_storeu_pd(&A[4], a_1_next);
        _mm256_storeu_pd(&A[8], a_2_next);

        // Increment pointers
        A += MR;
        C++;
        S++;
    }

    // Store final values of A
    _mm256_storeu_pd(&A[0], a_0_0);
    _mm256_storeu_pd(&A[4], a_1_0);
    _mm256_storeu_pd(&A[8], a_2_0);
}

void drotc_pack_A(int m, int n, const double * A, int lda, double * Ap){
    for(int ib = 0; ib + MR - 1 < m; ib+=MR){
        for(int j = 0; j < n; j++){
            __m256d a1 = _mm256_loadu_pd(&A[ib + j * lda]);
            __m256d a2 = _mm256_loadu_pd(&A[ib + j * lda + 4]);
            __m256d a3 = _mm256_loadu_pd(&A[ib + j * lda + 8]);
            _mm256_storeu_pd(&Ap[ib * n + j * MR], a1);
            _mm256_storeu_pd(&Ap[ib * n + j * MR + 4], a2);
            _mm256_storeu_pd(&Ap[ib * n + j * MR + 8], a3);
        }
    }

    int ib = m - m % MR;
    int m2 = m - ib;
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m2; i++){
            Ap[ib * n + j * MR + i] = A[i + ib + j * lda];
        }
    }

}

void drotc_unpack_A(int m, int n, const double * Ap, double * A, int lda){
    for(int ib = 0; ib + MR - 1 < m; ib+=MR){
        for(int j = 0; j < n; j++){
            __m256d a1 = _mm256_loadu_pd(&Ap[ib * n + j * MR]);
            __m256d a2 = _mm256_loadu_pd(&Ap[ib * n + j * MR + 4]);
            __m256d a3 = _mm256_loadu_pd(&Ap[ib * n + j * MR + 8]);
            _mm256_storeu_pd(&A[ib + j * lda], a1);
            _mm256_storeu_pd(&A[ib + j * lda + 4], a2);
            _mm256_storeu_pd(&A[ib + j * lda + 8], a3);
        }
    }

    int ib = m - m % MR;
    int m2 = m - ib;
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m2; i++){
            A[i + ib + j * lda] = Ap[ib * n + j * MR + i];
        }
    }
}