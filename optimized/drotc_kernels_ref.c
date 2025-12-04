#include "drotc_kernels.h"
#include "drotc_params.h"

void drotc_kernel_mrxnxkr(
    int n, double* A, const double* C, int ldc, const double* S, int lds)
{
    // Note, both MR and KR are 1 in this case, which simplifies the kernel
    for (int j = 0; j < n; j++) {
        double c = C[j];
        double s = S[j];
        double temp = c * A[j] + s * A[j + 1];
        A[j + 1] = -s * A[j] + c * A[j + 1];
        A[j] = temp;
    }
}

void drotc_kernel_mrxnx1(int n, double* A, const double* C, const double* S)
{
    for (int j = 0; j < n; j++) {
        double c = C[j];
        double s = S[j];
        double temp = c * A[j] + s * A[j + 1];
        A[j + 1] = -s * A[j] + c * A[j + 1];
        A[j] = temp;
    }
}

void drotc_kernel_mrxnxkr_backward(
    int n, double* A, const double* C, int ldc, const double* S, int lds)
{
    // Note, both MR and KR are 1 in this case, which simplifies the kernel
    for (int j = n-1; j >= 0; j--) {
        double c = C[j];
        double s = S[j];
        double temp = c * A[j] + s * A[j + 1];
        A[j + 1] = -s * A[j] + c * A[j + 1];
        A[j] = temp;
    }
}

void drotc_kernel_mrxnx1_backward(int n, double* A, const double* C, const double* S)
{
    for (int j = n-1; j >= 0; j--) {
        double c = C[j];
        double s = S[j];
        double temp = c * A[j] + s * A[j + 1];
        A[j + 1] = -s * A[j] + c * A[j + 1];
        A[j] = temp;
    }
}

void drotc_pack_A(int m, int n, const double* A, int lda, double* Ap)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            Ap[i * n + j] = A[i + j * lda];
        }
    }
}

void drotc_unpack_A(int m, int n, const double* Ap, double* A, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * lda] = Ap[i * n + j];
        }
    }
}