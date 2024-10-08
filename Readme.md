This repository contains a proposed extension to the Basic Linear Algebra Subroutines.

The proposed extension concerns Givens rotations. A rotation can be applied to two vectors using `rot`, but this is a level 1 BLAS operation and therefore its potential for efficient implementations is limited. The proposed extension add `rotc`, which applies a chain of sequences of rotations to a matrix. This is a level 3 BLAS operation and can be implemented efficiently on modern hardware, as demonstrated in our paper [1] and by the implementation in this repository.

Note that our efficient implementation does not offer all options. No application from the left, no backward application, no startup and no shutdown. We also only offer kernels optimized for AVX2.

[1] TODO: Add reference to paper