This repository contains a proposed extension to the Basic Linear Algebra Subroutines.

The proposed extension concerns Givens rotations. A rotation can be applied to two vectors using `rot`, but this is a level 1 BLAS operation and therefore its potential for efficient implementations is limited. The proposed extension add `rotc`, which applies a chain of sequences of rotations to a matrix. This is a level 3 BLAS operation and can be implemented efficiently on modern hardware, as demonstrated in our paper [1] and by the implementation in this repository.

Note that our efficient implementation does not offer all options. No application from the left, no backward application and the startup and shutdown phases are not fully optimized. The implementation is intended to demonstrate the potential of the operation and to provide a starting point for further development.

Below are some results of the performance of the implementation. We apply a chain of 180 sequences of rotations to a matrix of size n x n. The matrix is stored in column-major order. The performance is measured in terms of the number of flops per second. Rotations are inherently limited to 75% of the peak performance so the black line does not represent the peak performance of the hardware, but 75% of it.

![Plot of the performance of the implementation](./test/plot.pdf)

[1] TODO: Add reference to paper