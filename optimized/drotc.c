#include "drotc.h"

#include <stdlib.h>

#include "drotc_kernels.h"
#include "drotc_params.h"
#include "stdio.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

void drotc_startup_block(int m,
                         int n,
                         int k,
                         double* Ap,
                         int ldap,
                         const double* C,
                         int ldc,
                         const double* S,
                         int lds)
{
    for (int i = 0; i < m; i += MR) {
        for (int p = 0; p < k; p += 1) {
            // Note, we only use the mrxnx1 kernel here. This is generally less
            // efficient than using the mrxnxkr kernel, but it is simpler to
            // implement. We are assuming here that the startup phase is not
            // (that) performance critical.
            drotc_kernel_mrxnx1(n - p, &Ap[i * ldap], &C[p * ldc], &S[p * lds]);
        }
    }
}

void drotc_shutdown_block(int m,
                          int n,
                          int k,
                          double* Ap,
                          int ldap,
                          const double* C,
                          int ldc,
                          const double* S,
                          int lds)
{
    for (int i = 0; i < m; i += MR) {
        for (int p = 0, g = k - 1; p < k; p += 1, g -= 1) {
            // printf("p: %d, g: %d\n", p, g);
            // Note, we only use the mrxnx1 kernel here. This is generally less
            // efficient than using the mrxnxkr kernel, but it is simpler to
            // implement. We are assuming here that the shutdown phase is not
            // (that) performance critical.
            drotc_kernel_mrxnx1(n - g, &Ap[i * ldap + g * MR], &C[p * ldc + g],
                                &S[p * lds + g]);
        }
    }
}

void drotc_pipeline_block(int m,
                          int n,
                          int k,
                          double* Ap,
                          int ldap,
                          const double* C,
                          int ldc,
                          const double* S,
                          int lds)
{
    for (int i = 0; i < m; i += MR) {
        int p = 0;
        int g = k - 1;
        for (; p + (KR - 1) < k; p += KR, g -= KR) {
            drotc_kernel_mrxnxkr(n, &Ap[i * ldap + (g - (KR - 1)) * MR],
                                 &C[p * ldc], ldc, &S[p * lds], lds);
        }
        for (; p < k; p += 1, g -= 1) {
            drotc_kernel_mrxnx1(n, &Ap[i * ldap + g], &C[p * ldc], &S[p * lds]);
        }
    }
}

void drotc(char side,
           char dir,
           bool startup,
           bool shutdown,
           bool trans,
           int m,
           int n,
           int k,
           double* A,
           int lda,
           const double* C,
           int ldc,
           const double* S,
           int lds)
{
    if (side == 'L') {
        printf("Left rotation is not supported yet\n");
        return;
    }

    if (dir == 'B') {
        printf("Backward rotation is not supported yet\n");
        return;
    }

    if (trans) {
        printf("Transposed rotation is not supported yet\n");
        return;
    }

    // Make sure that kb and mb are multiples of kr and mr respectively
    const int nb = 216;
    const int kb = 60;
    const int mb = 960;

    const int m_pack = (min(m, mb) + MR - 1) / MR * MR;

    double* A_pack =
        (double*)aligned_alloc(64, m_pack * (n + 1) * sizeof(double));

    for (int ib = 0; ib < m; ib += mb) {
        int m2 = min(m - ib, mb);

        drotc_pack_A(m2, (n + 1), &A[ib], lda, A_pack);

        //
        // Startup phase
        //
        if (startup) {
            // Note, startup phase is more difficult to do with jb -> pb loop
            // order, so we do pb -> jb loop order here
            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);
                int jb = min(k - 1 - pb, k2);
                drotc_startup_block(m2, jb, k2, A_pack, n + 1, &C[pb * ldc],
                                    ldc, &S[pb * lds], lds);
                for (; jb < k - 1 - pb; jb += nb) {
                    int n2 = min(k - 1 - pb - jb, nb);
                    int gb = jb - (k2 - 1);
                    drotc_pipeline_block(m2, n2, k2, &A_pack[gb * MR], n + 1,
                                         &C[pb * ldc + jb], ldc - 1,
                                         &S[pb * lds + jb], lds - 1);
                }
            }
        }

        //
        // Pipeline phase
        //
        for (int jb = 0; jb < n - k + 1; jb += nb) {
            int n2 = min(n - k + 1 - jb, nb);

            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);

                int gb = jb + k - 1 - pb;
                int gb2 = gb - (k2 - 1);

                // printf("ib: %d, jb: %d, pb: %d\n", ib, jb, pb);
                // printf("gb: %d, gb2: %d\n", gb, gb2);
                // printf("m2: %d, n2: %d, k2: %d\n", m2, n2, k2);

                drotc_pipeline_block(m2, n2, k2, &A_pack[gb2 * MR], n + 1,
                                     &C[gb + pb * ldc], ldc - 1,
                                     &S[gb + pb * lds], lds - 1);
            }
        }

        //
        // Shutdown phase
        //
        if (shutdown) {
            // Note, shutdown phase is more difficult to do with jb -> pb loop
            // order, so we do pb -> jb loop order here
            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);
                int jb = n - pb - (k2 - 1);
                int n2 = n - jb;
                drotc_shutdown_block(m2, n2, k2, &A_pack[jb * MR], n + 1,
                                     &C[pb * ldc + jb], ldc, &S[pb * lds + jb],
                                     lds);
            }
        }

        drotc_unpack_A(m2, (n + 1), A_pack, &A[ib], lda);
    }

    free(A_pack);
}

void drotc_prepacked(char side,
                     char dir,
                     bool startup,
                     bool shutdown,
                     bool trans,
                     int m,
                     int n,
                     int k,
                     double* A_pack,
                     int ldap,
                     const double* C,
                     int ldc,
                     const double* S,
                     int lds)
{
    if (side == 'L') {
        printf("Left rotation is not supported yet\n");
        return;
    }

    if (dir == 'B') {
        printf("Backward rotation is not supported yet\n");
        return;
    }

    if (trans) {
        printf("Transposed rotation is not supported yet\n");
        return;
    }

    // Make sure that kb and mb are multiples of kr and mr respectively
    const int nb = 216;
    const int kb = 60;
    const int mb = 960;

    for (int ib = 0; ib < m; ib += mb) {
        int m2 = min(m - ib, mb);

        //
        // Startup phase
        //
        if (startup) {
            // Note, startup phase is more difficult to do with jb -> pb loop
            // order, so we do pb -> jb loop order here
            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);
                int jb = min(k - 1 - pb, k2);
                drotc_startup_block(m2, jb, k2, A_pack, n + 1, &C[pb * ldc],
                                    ldc, &S[pb * lds], lds);
                for (; jb < k - 1 - pb; jb += nb) {
                    int n2 = min(k - 1 - pb - jb, nb);
                    int gb = jb - (k2 - 1);
                    drotc_pipeline_block(m2, n2, k2, &A_pack[gb * MR], n + 1,
                                         &C[pb * ldc + jb], ldc - 1,
                                         &S[pb * lds + jb], lds - 1);
                }
            }
        }

        //
        // Pipeline phase
        //
        for (int jb = 0; jb < n - k + 1; jb += nb) {
            int n2 = min(n - k + 1 - jb, nb);

            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);

                int gb = jb + k - 1 - pb;
                int gb2 = gb - (k2 - 1);

                // printf("ib: %d, jb: %d, pb: %d\n", ib, jb, pb);
                // printf("gb: %d, gb2: %d\n", gb, gb2);
                // printf("m2: %d, n2: %d, k2: %d\n", m2, n2, k2);

                drotc_pipeline_block(m2, n2, k2, &A_pack[gb2 * MR], n + 1,
                                     &C[gb + pb * ldc], ldc - 1,
                                     &S[gb + pb * lds], lds - 1);
            }
        }

        //
        // Shutdown phase
        //
        if (shutdown) {
            // Note, shutdown phase is more difficult to do with jb -> pb loop
            // order, so we do pb -> jb loop order here
            for (int pb = 0; pb < k; pb += kb) {
                int k2 = min(k - pb, kb);
                int jb = n - pb - (k2 - 1);
                int n2 = n - jb;
                drotc_shutdown_block(m2, n2, k2, &A_pack[jb * MR], n + 1,
                                     &C[pb * ldc + jb], ldc, &S[pb * lds + jb],
                                     lds);
            }
        }
    }
}