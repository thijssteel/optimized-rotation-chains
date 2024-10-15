#include "drotc_params.h"
#include "drotc_kernels.h"

#include <immintrin.h>

void drotc_kernel_mrxnxkr(int n, double * A, const double * C, int ldc, const double * S, int lds)
{
    // Load initial values of A
    __m512d a_0_0 = _mm512_loadu_pd(A);
    __m512d a_1_0 = _mm512_loadu_pd(&A[8]);
    __m512d a_2_0 = _mm512_loadu_pd(&A[16]);
    __m512d a_3_0 = _mm512_loadu_pd(&A[24]);

    __m512d a_0_1 = _mm512_loadu_pd(&A[MR]);
    __m512d a_1_1 = _mm512_loadu_pd(&A[MR + 8]);
    __m512d a_2_1 = _mm512_loadu_pd(&A[MR + 16]);
    __m512d a_3_1 = _mm512_loadu_pd(&A[MR + 24]);

    __m512d a_0_2 = _mm512_loadu_pd(&A[2 * MR]);
    __m512d a_1_2 = _mm512_loadu_pd(&A[2 * MR + 8]);
    __m512d a_2_2 = _mm512_loadu_pd(&A[2 * MR + 16]);
    __m512d a_3_2 = _mm512_loadu_pd(&A[2 * MR + 24]);

    __m512d a_0_3 = _mm512_loadu_pd(&A[3 * MR]);
    __m512d a_1_3 = _mm512_loadu_pd(&A[3 * MR + 8]);
    __m512d a_2_3 = _mm512_loadu_pd(&A[3 * MR + 16]);
    __m512d a_3_3 = _mm512_loadu_pd(&A[3 * MR + 24]);

    __m512d a_0_4 = _mm512_loadu_pd(&A[4 * MR]);
    __m512d a_1_4 = _mm512_loadu_pd(&A[4 * MR + 8]);
    __m512d a_2_4 = _mm512_loadu_pd(&A[4 * MR + 16]);
    __m512d a_3_4 = _mm512_loadu_pd(&A[4 * MR + 24]);

    __m512d a_0_5 = _mm512_loadu_pd(&A[5 * MR]);
    __m512d a_1_5 = _mm512_loadu_pd(&A[5 * MR + 8]);
    __m512d a_2_5 = _mm512_loadu_pd(&A[5 * MR + 16]);
    __m512d a_3_5 = _mm512_loadu_pd(&A[5 * MR + 24]);

    for (int i = 0; i < n; i++)
    {
        __m512d temp, c_vec, s_vec;

        // Load new values of A
        __m512d a_0_next = _mm512_loadu_pd(&A[6 * MR]);
        __m512d a_1_next = _mm512_loadu_pd(&A[6 * MR + 8]);
        __m512d a_2_next = _mm512_loadu_pd(&A[6 * MR + 16]);
        __m512d a_3_next = _mm512_loadu_pd(&A[6 * MR + 24]);

        // Apply rotation 0 to values 5 and 6 (next)
        c_vec = _mm512_set1_pd(C[0]);
        s_vec = _mm512_set1_pd(S[0]);

        temp = _mm512_fmadd_pd(c_vec, a_0_5, _mm512_mul_pd(s_vec, a_0_next));
        a_0_5 = _mm512_fnmadd_pd(s_vec, a_0_5, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_5, _mm512_mul_pd(s_vec, a_1_next));
        a_1_5 = _mm512_fnmadd_pd(s_vec, a_1_5, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_5, _mm512_mul_pd(s_vec, a_2_next));
        a_2_5 = _mm512_fnmadd_pd(s_vec, a_2_5, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_5, _mm512_mul_pd(s_vec, a_3_next));
        a_3_5 = _mm512_fnmadd_pd(s_vec, a_3_5, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Apply rotation 1 to values 4 and 5 (next)
        c_vec = _mm512_set1_pd(C[ldc]);
        s_vec = _mm512_set1_pd(S[lds]);

        temp = _mm512_fmadd_pd(c_vec, a_0_4, _mm512_mul_pd(s_vec, a_0_next));
        a_0_4 = _mm512_fnmadd_pd(s_vec, a_0_4, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_4, _mm512_mul_pd(s_vec, a_1_next));
        a_1_4 = _mm512_fnmadd_pd(s_vec, a_1_4, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_4, _mm512_mul_pd(s_vec, a_2_next));
        a_2_4 = _mm512_fnmadd_pd(s_vec, a_2_4, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_4, _mm512_mul_pd(s_vec, a_3_next));
        a_3_4 = _mm512_fnmadd_pd(s_vec, a_3_4, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Apply rotation 2 to values 3 and 4 (next)
        c_vec = _mm512_set1_pd(C[2 * ldc]);
        s_vec = _mm512_set1_pd(S[2 * lds]);

        temp = _mm512_fmadd_pd(c_vec, a_0_3, _mm512_mul_pd(s_vec, a_0_next));
        a_0_3 = _mm512_fnmadd_pd(s_vec, a_0_3, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_3, _mm512_mul_pd(s_vec, a_1_next));
        a_1_3 = _mm512_fnmadd_pd(s_vec, a_1_3, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_3, _mm512_mul_pd(s_vec, a_2_next));
        a_2_3 = _mm512_fnmadd_pd(s_vec, a_2_3, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_3, _mm512_mul_pd(s_vec, a_3_next));
        a_3_3 = _mm512_fnmadd_pd(s_vec, a_3_3, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Apply rotation 3 to values 2 and 3 (next)
        c_vec = _mm512_set1_pd(C[3 * ldc]);
        s_vec = _mm512_set1_pd(S[3 * lds]);

        temp = _mm512_fmadd_pd(c_vec, a_0_2, _mm512_mul_pd(s_vec, a_0_next));
        a_0_2 = _mm512_fnmadd_pd(s_vec, a_0_2, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_2, _mm512_mul_pd(s_vec, a_1_next));
        a_1_2 = _mm512_fnmadd_pd(s_vec, a_1_2, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_2, _mm512_mul_pd(s_vec, a_2_next));
        a_2_2 = _mm512_fnmadd_pd(s_vec, a_2_2, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_2, _mm512_mul_pd(s_vec, a_3_next));
        a_3_2 = _mm512_fnmadd_pd(s_vec, a_3_2, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Apply rotation 4 to values 1 and 2 (next)
        c_vec = _mm512_set1_pd(C[4 * ldc]);
        s_vec = _mm512_set1_pd(S[4 * lds]);

        temp = _mm512_fmadd_pd(c_vec, a_0_1, _mm512_mul_pd(s_vec, a_0_next));
        a_0_1 = _mm512_fnmadd_pd(s_vec, a_0_1, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_1, _mm512_mul_pd(s_vec, a_1_next));
        a_1_1 = _mm512_fnmadd_pd(s_vec, a_1_1, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_1, _mm512_mul_pd(s_vec, a_2_next));
        a_2_1 = _mm512_fnmadd_pd(s_vec, a_2_1, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_1, _mm512_mul_pd(s_vec, a_3_next));
        a_3_1 = _mm512_fnmadd_pd(s_vec, a_3_1, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Apply rotation 5 to values 0 and 1 (next)
        c_vec = _mm512_set1_pd(C[5 * ldc]);
        s_vec = _mm512_set1_pd(S[5 * lds]);

        temp = _mm512_fmadd_pd(c_vec, a_0_0, _mm512_mul_pd(s_vec, a_0_next));
        a_0_0 = _mm512_fnmadd_pd(s_vec, a_0_0, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_1_0, _mm512_mul_pd(s_vec, a_1_next));
        a_1_0 = _mm512_fnmadd_pd(s_vec, a_1_0, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_2_0, _mm512_mul_pd(s_vec, a_2_next));
        a_2_0 = _mm512_fnmadd_pd(s_vec, a_2_0, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp;

        temp = _mm512_fmadd_pd(c_vec, a_3_0, _mm512_mul_pd(s_vec, a_3_next));
        a_3_0 = _mm512_fnmadd_pd(s_vec, a_3_0, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp;

        // Store value 0 (stored in next)
        _mm512_storeu_pd(&A[0], a_0_next);
        _mm512_storeu_pd(&A[8], a_1_next);
        _mm512_storeu_pd(&A[16], a_2_next);
        _mm512_storeu_pd(&A[24], a_3_next);

        // Increment pointers
        A += MR;
        C++;
        S++;
    }

    // Store final values of A
    _mm512_storeu_pd(&A[0], a_0_0);
    _mm512_storeu_pd(&A[8], a_1_0);
    _mm512_storeu_pd(&A[16], a_2_0);
    _mm512_storeu_pd(&A[24], a_3_0);

    _mm512_storeu_pd(&A[MR + 0], a_0_1);
    _mm512_storeu_pd(&A[MR + 8], a_1_1);
    _mm512_storeu_pd(&A[MR + 16], a_2_1);
    _mm512_storeu_pd(&A[MR + 24], a_3_1);

    _mm512_storeu_pd(&A[2 * MR + 0], a_0_2);
    _mm512_storeu_pd(&A[2 * MR + 8], a_1_2);
    _mm512_storeu_pd(&A[2 * MR + 16], a_2_2);
    _mm512_storeu_pd(&A[2 * MR + 24], a_3_2);

    _mm512_storeu_pd(&A[3 * MR + 0], a_0_3);
    _mm512_storeu_pd(&A[3 * MR + 8], a_1_3);
    _mm512_storeu_pd(&A[3 * MR + 16], a_2_3);
    _mm512_storeu_pd(&A[3 * MR + 24], a_3_3);

    _mm512_storeu_pd(&A[4 * MR + 0], a_0_4);
    _mm512_storeu_pd(&A[4 * MR + 8], a_1_4);
    _mm512_storeu_pd(&A[4 * MR + 16], a_2_4);
    _mm512_storeu_pd(&A[4 * MR + 24], a_3_4);

    _mm512_storeu_pd(&A[5 * MR + 0], a_0_5);
    _mm512_storeu_pd(&A[5 * MR + 8], a_1_5);
    _mm512_storeu_pd(&A[5 * MR + 16], a_2_5);
    _mm512_storeu_pd(&A[5 * MR + 24], a_3_5);
}

void drotc_kernel_mrxnx1(int n, double * A, const double * C, const double * S)
{
    // Load initial values of A
    __m512d a_0_0 = _mm512_loadu_pd(A);
    __m512d a_1_0 = _mm512_loadu_pd(&A[8]);
    __m512d a_2_0 = _mm512_loadu_pd(&A[16]);
    __m512d a_3_0 = _mm512_loadu_pd(&A[24]);

    for (int i = 0; i < n; i++)
    {
        __m512d temp1, temp2, temp3, temp4, c_vec, s_vec;

        // Load new values of A
        __m512d a_0_next = _mm512_loadu_pd(&A[MR]);
        __m512d a_1_next = _mm512_loadu_pd(&A[MR + 8]);
        __m512d a_2_next = _mm512_loadu_pd(&A[MR + 16]);
        __m512d a_3_next = _mm512_loadu_pd(&A[MR + 24]);

        // Apply rotation 0 to values 0 and 1 (next)
        c_vec = _mm512_set1_pd(C[0]);
        s_vec = _mm512_set1_pd(S[0]);

        temp1 = _mm512_fmadd_pd(c_vec, a_0_0, _mm512_mul_pd(s_vec, a_0_next));
        a_0_0 = _mm512_fnmadd_pd(s_vec, a_0_0, _mm512_mul_pd(c_vec, a_0_next));
        a_0_next = temp1;

        temp2 = _mm512_fmadd_pd(c_vec, a_1_0, _mm512_mul_pd(s_vec, a_1_next));
        a_1_0 = _mm512_fnmadd_pd(s_vec, a_1_0, _mm512_mul_pd(c_vec, a_1_next));
        a_1_next = temp2;

        temp3 = _mm512_fmadd_pd(c_vec, a_2_0, _mm512_mul_pd(s_vec, a_2_next));
        a_2_0 = _mm512_fnmadd_pd(s_vec, a_2_0, _mm512_mul_pd(c_vec, a_2_next));
        a_2_next = temp3;

        temp4 = _mm512_fmadd_pd(c_vec, a_3_0, _mm512_mul_pd(s_vec, a_3_next));
        a_3_0 = _mm512_fnmadd_pd(s_vec, a_3_0, _mm512_mul_pd(c_vec, a_3_next));
        a_3_next = temp4;

        // Store value 0 (stored in next)
        _mm512_storeu_pd(&A[0], a_0_next);
        _mm512_storeu_pd(&A[8], a_1_next);
        _mm512_storeu_pd(&A[16], a_2_next);
        _mm512_storeu_pd(&A[24], a_3_next);

        // Increment pointers
        A += MR;
        C++;
        S++;
    }

    // Store final values of A
    _mm512_storeu_pd(&A[0], a_0_0);
    _mm512_storeu_pd(&A[8], a_1_0);
    _mm512_storeu_pd(&A[16], a_2_0);
    _mm512_storeu_pd(&A[24], a_3_0);
}

void drotc_pack_A(int m, int n, const double * A, int lda, double * Ap){
    for(int ib = 0; ib + MR - 1 < m; ib+=MR){
        for(int j = 0; j < n; j++){
            __m512d a1 = _mm512_loadu_pd(&A[ib + j * lda]);
            __m512d a2 = _mm512_loadu_pd(&A[ib + j * lda + 8]);
            __m512d a3 = _mm512_loadu_pd(&A[ib + j * lda + 16]);
            __m512d a4 = _mm512_loadu_pd(&A[ib + j * lda + 24]);
            _mm512_storeu_pd(&Ap[ib * n + j * MR], a1);
            _mm512_storeu_pd(&Ap[ib * n + j * MR + 8], a2);
            _mm512_storeu_pd(&Ap[ib * n + j * MR + 16], a3);
            _mm512_storeu_pd(&Ap[ib * n + j * MR + 24], a4);
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
            __m512d a1 = _mm512_loadu_pd(&Ap[ib * n + j * MR]);
            __m512d a2 = _mm512_loadu_pd(&Ap[ib * n + j * MR + 8]);
            __m512d a3 = _mm512_loadu_pd(&Ap[ib * n + j * MR + 16]);
            __m512d a4 = _mm512_loadu_pd(&Ap[ib * n + j * MR + 24]);
            _mm512_storeu_pd(&A[ib + j * lda], a1);
            _mm512_storeu_pd(&A[ib + j * lda + 8], a2);
            _mm512_storeu_pd(&A[ib + j * lda + 16], a3);
            _mm512_storeu_pd(&A[ib + j * lda + 24], a4);
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