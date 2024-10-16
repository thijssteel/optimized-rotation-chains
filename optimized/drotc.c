#include "drotc.h"
#include "drotc_params.h"
#include "drotc_kernels.h"

#include <stdlib.h>
#include "stdio.h"


#define min(a, b) ((a) < (b) ? (a) : (b))

void drotc_pipeline_block(int m, int n, int k, double *Ap, int ldap, const double *C, int ldc, const double *S, int lds)
{

    for (int i = 0; i < m; i += MR)
    {
        int p = 0;
        int g = k - 1;
        for (; p + (KR - 1) < k; p += KR, g -= KR)
        {
            drotc_kernel_mrxnxkr(n, &Ap[i * ldap + (g - (KR - 1)) * MR], &C[p * ldc], ldc, &S[p * lds], lds);
        }
        for (; p < k; p += 1, g -= 1)
        {
            drotc_kernel_mrxnx1(n, &Ap[i * ldap + g], &C[p * ldc], &S[p * lds]);
        }
    }
}

void drotc(char side, char dir, bool startup, bool shutdown,
           int m, int n, int k, double *A, int lda,
           const double *C, int ldc, const double *S, int lds)
{

    if (side == 'L')
    {
        printf("Left rotation is not supported yet\n");
        return;
    }

    if (dir == 'B')
    {
        printf("Backward rotation is not supported yet\n");
        return;
    }

    if (startup)
    {
        printf("Startup rotation is not supported yet\n");
        return;
    }

    if (shutdown)
    {
        printf("Shutdown rotation is not supported yet\n");
        return;
    }

    // Make sure that kb and mb are multiples of kr and mr respectively
    const int nb = 216;
    const int kb = 60;
    const int mb = 960;

    const int m_pack = (min(m, mb) + MR - 1) / MR * MR;

    double *A_pack =
        (double *)aligned_alloc(64, m_pack * (n+1) * sizeof(double));

    
    for (int ib = 0; ib < m; ib += mb)
    {
        int m2 = min(m - ib, mb);

        drotc_pack_A(m2, (n+1), &A[ib], lda, A_pack);

        for (int jb = 0; jb < n - k + 1; jb += nb)
        {
            int n2 = min(n - k + 1 - jb, nb);

            for (int pb = 0; pb < k; pb += kb)
            {
                int k2 = min(k - pb, kb);

                int gb = jb + k - 1 - pb;
                int gb2 = gb - (k2 - 1);

                // printf("ib: %d, jb: %d, pb: %d\n", ib, jb, pb);
                // printf("gb: %d, gb2: %d\n", gb, gb2);
                // printf("m2: %d, n2: %d, k2: %d\n", m2, n2, k2);

                drotc_pipeline_block(m2, n2, k2, &A_pack[gb2*MR], n+1, &C[gb + pb * ldc],
                                           ldc - 1, &S[gb + pb * lds], lds - 1);
            }
        }

        drotc_unpack_A(m2, (n+1), A_pack, &A[ib], lda);
    }

    free(A_pack);
}

void drotc_prepacked(char side, char dir, bool startup, bool shutdown,
           int m, int n, int k, double *A_pack, int ldap,
           const double *C, int ldc, const double *S, int lds)
{

    if (side == 'L')
    {
        printf("Left rotation is not supported yet\n");
        return;
    }

    if (dir == 'B')
    {
        printf("Backward rotation is not supported yet\n");
        return;
    }

    if (startup)
    {
        printf("Startup rotation is not supported yet\n");
        return;
    }

    if (shutdown)
    {
        printf("Shutdown rotation is not supported yet\n");
        return;
    }

    // Make sure that kb and mb are multiples of kr and mr respectively
    const int nb = 216;
    const int kb = 60;
    const int mb = 960;
    
    for (int ib = 0; ib < m; ib += mb)
    {
        int m2 = min(m - ib, mb);

        for (int jb = 0; jb < n - k + 1; jb += nb)
        {
            int n2 = min(n - k + 1 - jb, nb);

            for (int pb = 0; pb < k; pb += kb)
            {
                int k2 = min(k - pb, kb);

                int gb = jb + k - 1 - pb;
                int gb2 = gb - (k2 - 1);

                drotc_pipeline_block(m2, n2, k2, &A_pack[gb2*MR], ldap, &C[gb + pb * ldc],
                                           ldc - 1, &S[gb + pb * lds], lds - 1);
            }
        }
    }
}