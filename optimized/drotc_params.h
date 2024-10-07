#ifndef DROTC_PARAMS_H
#define DROTC_PARAMS_H

#define MR 1
#define KR 1

#ifdef __AVX__
#undef MR
#undef KR
#define MR 12
#define KR 3
#endif

#ifdef __AVX2__
#undef MR
#undef KR
#define MR 12
#define KR 3
#endif

#ifdef __AVX512F__
#undef MR
#undef KR
#define MR 24
#define KR 6
#endif

#endif