#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(const nn_real* A, const nn_real* B, nn_real* C, nn_real* alpha, nn_real* beta, int M, int N,
           int K);

int myGEMMOverwrite(const nn_real* A, const nn_real* B, nn_real* C, nn_real* alpha, int M, int N,
           int K);

int myGEMM(const nn_real* A, const nn_real* B, const nn_real* C, nn_real* out, nn_real* alpha, nn_real* beta, int M, int N,
           int K);

void sigmoid_gpu(const nn_real* in, nn_real* out, int N);

void softmax_gpu(const nn_real* A, nn_real* out, int M, int N);


nn_real ce_gpu(const nn_real* yc, const nn_real* y, int M, int N);

void gpu_norm(const nn_real* w, nn_real* out, int N);

void sum(const nn_real* a, nn_real* out, int N) ;

void transpose(const nn_real* a, nn_real* out, int M, int N);

void sigmoid_derivative(const nn_real* a, const nn_real* b, nn_real* out, int N);

void rowwise_sum(const nn_real* a, nn_real* out, int M, int N);

void scaled_difference(const nn_real* a, const nn_real* b, nn_real* out, int M, int N);

void gradient_update(const nn_real* grad, nn_real* weights, nn_real scalar, int N);

// broad cast a column vector b of size M, N times in along axis 1.
void broadcast_rowwise(const nn_real* b, nn_real* out, int M, int N);


int myGEMM_SLOW(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K);
size_t free_memory_on_gpu();
// TODO
// Add additional function declarations


#endif
