#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"



/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

// A note that this code was adapted for this example from online but I 
// made sure to understand it.
__global__ 
void reduce_sum(const nn_real* arr, nn_real* out, int N) {

    int idx = threadIdx.x;
    nn_real sum = 0;
    for (int i = idx; i < N; i += 1024)
        sum += arr[i];
    __shared__ nn_real block[1024];
    block[idx] = sum;
    __syncthreads();
    for (int size =  1024 / 2; size>0; size/=2) { //uniform
        if (idx<size)
            block[idx] += block[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        *out = block[0];
}

__global__
void cross_entropy_no_reduce_k(const nn_real* __restrict__ yc, const nn_real* __restrict__ y, int M, int N, nn_real* out) {
  int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
  if (y_idx < M && x_idx < N) {
      nn_real c = y[y_idx + M * x_idx];
      if (c > 0.99) { // therefore 1.
        out[x_idx] = -log(yc[x_idx * M + y_idx]);
      }
  }
}

__global__ void square_k(const nn_real* wc, nn_real* w_square, int N) {
  int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (x_idx < N) {
    w_square[x_idx] = pow(wc[x_idx], 2.0);
  }
}


__global__
void rowwise_sum_kernel(const nn_real* in, nn_real* out, int M, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < M) {
    nn_real sum = 0; // TODO Pre compute this.
    for(int i = 0; i < N; i++) {
      sum += in[x + M * i];
    }

    out[x] =  sum;
  }
}


__global__
void softmax_k(const nn_real* in, nn_real* out, int M, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (y < M && x < N) {
    nn_real sum = 0.0; // TODO Pre compute this.
    for(int i = 0; i < M; i++) {
      sum += exp(in[i + M * x]);
    }

    out[y + M * x] = exp(in[y + M * x]) / sum;
  }
}

__global__
void sigmoid_k(const nn_real* in, nn_real* out, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < N) {
    nn_real element = in[x];
    nn_real val = exp(-element);
    out[x] = (nn_real) 1.0 / ((nn_real) 1.0 + val);
  }
}

__global__
void __tile_gemm(const nn_real* A, const nn_real*  B, nn_real*  C, nn_real alpha, nn_real beta, int M, int N, int K) {

  int globalOffsetX = blockDim.x * blockIdx.x; // blockDim.x = 16
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * blockDim.y;

  int k = 0;

  int row_a = blockIdx.y * num_threads + thread_id;
  int col_B = globalOffsetX + threadIdx.x;

  int n_B_rows = K;
  int n_B_cols = N;
  int c_n_rows = M;
  // if (row_a < M) {
  //   for(int i = 0; i < blockDim.x; i++) {
  //     int c_col = (globalOffsetX + i);
  //     if (c_col < N) {
  //       C[row_a + c_n_rows * c_col] = 0; // zero it out.
  //     }
  //   }
  // }

  // __syncthreads();
  // Iterate across A's cols.
  nn_real out[16];
  for(int i = 0; i < 16; i++) {
    out[i] = 0;
  }
  // out[threadIdx.x] = 0;
  // __syncthreads();
  

  while(k < K) {

    // if(thread_id == 0) {
    //   printf("k: %i\n", k);
    // }

    __shared__ nn_real b_block[4][16];
    int row_B = k + threadIdx.y; // threadIdx.y : [0..<k_increment]
  
    if (row_B < n_B_rows && col_B < n_B_cols) {
      b_block[threadIdx.y][threadIdx.x] = B[row_B + n_B_rows * col_B];
    } else {
      b_block[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();


    if (row_a >= M) {
      return;
    }

  
    for(int i = 0; i < blockDim.x; i++) {
      nn_real dot_prod = 0;
      
      for(int j = 0; j < blockDim.y; j++) { 
          int col_a = k + j;

          if (col_a >= K) {
            continue;
          }

          int a_index = row_a + M * col_a;
          nn_real b = b_block[j][i];
          nn_real a = A[a_index];
          
          dot_prod += b * a;
      }

      int c_col = (globalOffsetX + i);
      if (c_col >= N)  {

        out[i] += dot_prod;
        break;
      } else {
        out[i] += dot_prod;
        // C[row_a + c_n_rows * c_col] += dot_prod * alpha;
      }
    }
    k += blockDim.y;
    __syncthreads();

  }

  for(int i = 0; i < 16; i++) {
    int c_col = (globalOffsetX + i);
    C[row_a + c_n_rows * c_col] = C[row_a + c_n_rows * c_col] * beta + out[i] * alpha;
  }
}


__global__
void gemm_kernel_external_mem(const nn_real* A, const nn_real*  B, const nn_real*  C, nn_real* out, nn_real alpha, nn_real beta, int M, int N, int K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (y >= M || x >= N) {
    return;
  }

  nn_real beta_c = beta * C[M * x + y];
  nn_real mat_mul_sum = 0.0;

  // Fix the ith row of A, fix the jth col of B.
  // Do dot product over A[i, :] and B[:, j].
  for(int k = 0; k < K; k++) {
    // Grab the yth row of A. In column major order, we need to
    // step by the number of rows (M for A, K for B) each time.
    nn_real a = A[y + M * k];

    // Grab the xth column of B. A column is represented contiguously
    // in memory so just iterate by k steps and watch the offset.
    nn_real b = B[k + K * x];
    mat_mul_sum += a * b;
  } 
  
  out[M * x + y] = mat_mul_sum * alpha + beta_c;
}


__global__
void mm_kernel_slow(const nn_real* A, const nn_real*  B, nn_real* out, nn_real alpha, nn_real beta, int M, int N, int K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (y >= M || x >= N) {
    return;
  }

  // nn_real beta_c = beta * C[M * x + y];
  nn_real mat_mul_sum = 0.0;

  // Fix the ith row of A, fix the jth col of B.
  // Do dot product over A[i, :] and B[:, j].
  for(int k = 0; k < K; k++) {
    // Grab the yth row of A. In column major order, we need to
    // step by the number of rows (M for A, K for B) each time.
    nn_real a = A[y + M * k];

    // Grab the xth column of B. A column is represented contiguously
    // in memory so just iterate by k steps and watch the offset.
    nn_real b = B[k + K * x];
    mat_mul_sum += a * b;

  } 
  int index = M * x + y;
  out[index] =  mat_mul_sum * alpha + beta * out[index];
}
__global__
void mm_kernel_external_mem(const nn_real* A, const nn_real*  B, nn_real* out, nn_real alpha, int M, int N, int K) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (y >= M || x >= N) {
    return;
  }



  // nn_real beta_c = beta * C[M * x + y];
  nn_real mat_mul_sum = 0.0;

  // Fix the ith row of A, fix the jth col of B.
  // Do dot product over A[i, :] and B[:, j].
  for(int k = 0; k < K; k++) {
    // Grab the yth row of A. In column major order, we need to
    // step by the number of rows (M for A, K for B) each time.
    nn_real a = A[y + M * k];

    // Grab the xth column of B. A column is represented contiguously
    // in memory so just iterate by k steps and watch the offset.
    nn_real b = B[k + K * x];
    mat_mul_sum += a * b;
  } 
  
  out[M * x + y] = mat_mul_sum * alpha;
}

int myGEMMOverwrite(const nn_real* A, const nn_real* B, nn_real* C, nn_real* alpha, int M, int N,
           int K) {
    dim3 threads(32, 32);
    int grid_y = (M + 32 - 1) / 32;
    int grid_x = (N + 32 - 1) / 32;
    dim3 blocks(grid_x, grid_y);
    mm_kernel_external_mem<<<blocks, threads>>>(A, B, C, *alpha, M, N, K);
    return 0;
}

__global__ void gemm_simple(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real alpha, nn_real beta,
           int M, int N, int K) {

  nn_real CValue = 0.0;

  // NEED TO DO ONE MORE BLOCK THAN K / 32 in some cases.
  int blocks = (K + 32 - 1) / 32;

  int c_row = blockIdx.y * 32 + threadIdx.y;
  int c_col = blockIdx.x * 32 + threadIdx.x;

  for (int m = 0; m < blocks; ++m) {
    __shared__ nn_real A_shared[32][32];
    __shared__ nn_real B_shared[32][32];
    int a_col = (m * 32 + threadIdx.x);
    int b_row = (m * 32 + threadIdx.y);

    int n_A_rows = M;
    int n_B_rows = K;
    int n_A_cols = K;
    int n_C_rows = M;
    int n_C_cols = N;

  // THESE NEED TO BE NOT MUTUALLY EXCLUSIVE!
    if (a_col >= n_A_cols) {
      A_shared[threadIdx.y][threadIdx.x] = 0.0;
    } else if (c_row >= n_C_rows) {
      A_shared[threadIdx.y][threadIdx.x] = 0.0;
    } else {
            // Load A[blockIdx.y : blockIdx.y + 32, m * 32 : (m + 1) * 32] into shared.
      A_shared[threadIdx.y][threadIdx.x] = A[a_col * n_A_rows + c_row];
    }
    
    if (b_row >= n_B_rows) {
      B_shared[threadIdx.y][threadIdx.x] = 0.0;
    } else if (c_col >= n_C_cols) {
      B_shared[threadIdx.y][threadIdx.x] = 0.0;
    } else {
      // Load B[m * 32 : (m + 1) * 32, blockIdx.x : blockIdx.x + 32] into shared.
      B_shared[threadIdx.y][threadIdx.x] = B[c_col * n_B_rows + b_row];
    }

    __syncthreads();

    for(int s = 0; s < 32; ++s) {
      CValue += A_shared[threadIdx.y][s] * B_shared[s][threadIdx.x];
    }

    __syncthreads();
  }

  int c_index = c_row + M * c_col;

  if(c_row >= M) {
    return;
  }

  if(c_col >= N) {
    return;
  }
  C[c_index] = CValue * alpha + beta * C[c_index];     
}


int myGEMM(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
                 dim3 threads(32, 32);
            int grid_y = (M + 32 - 1) / 32;
            int grid_x = (N + 32 - 1) / 32;
            dim3 blocks(grid_x, grid_y);
            gemm_simple<<<blocks, threads>>>(A, B, C,  *alpha, *beta, M, N, K);
            // check_launch("gemm_ext");
            return 0;
  

             return 0;
}

/*
  Caller function GEMM
*/
int myGEMM_SLOW(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
    // std::cout << "new gemm" << std::endl;

    int blockSizeX = 32;
    int blockSizeY = 32;
    dim3 threads(blockSizeX, blockSizeY);

    int grid_y = (M + blockSizeY - 1) / blockSizeY;
    int grid_x = (N + 32 - 1) / 32;
    dim3 blocks(grid_x, grid_y);

    mm_kernel_slow<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    
    
    check_launch("gemm");
    return 0;
}

int myGEMM_attempt(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
    // std::cout << "new gemm" << std::endl;

    int blockSizeX = 16;
    int blockSizeY = 4;
    dim3 threads(blockSizeX, blockSizeY);

    int grid_y = (M + blockSizeY - 1) / blockSizeY;
    int grid_x = (N + blockSizeX - 1) / blockSizeX;
    dim3 blocks(grid_x, grid_y);

    __tile_gemm<<<blocks, threads>>>(A, B, C, *alpha, *beta, M, N, K);
    
    
    check_launch("gemm");
    return 0;
}




int myGEMM(const nn_real* __restrict__ A, const nn_real* __restrict__ B,
           const nn_real* __restrict__ C, nn_real* out, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {
    dim3 threads(32, 32);
    int grid_y = (M + 32 - 1) / 32;
    int grid_x = (N + 32 - 1) / 32;
    dim3 blocks(grid_x, grid_y);
    gemm_kernel_external_mem<<<blocks, threads>>>(A, B, C, out, *alpha, *beta, M, N, K);
    // check_launch("gemm_ext");
    return 0;
}

void sigmoid_gpu(const nn_real* __restrict__ in, nn_real* __restrict__ out, int N) {
  int blocks = (N + 1024 - 1) / 1024;
  sigmoid_k<<<blocks, 1024>>>(in, out, N);
  // check_launch("sigmoid_gpu");
}


void softmax_gpu(const nn_real* __restrict__ A, nn_real* __restrict__ out, int M, int N) {
  dim3 threads(32, 32);
  int grid_x = (N * 32 - 1) / 32;
  int grid_y = (M * 32 - 1) / 32;
  dim3 blocks(grid_x, grid_y);
  softmax_k<<<blocks, threads>>>(A, out, M, N);
  // check_launch("softmax_gpu");
}

nn_real ce_gpu(const nn_real* yc, const nn_real* y, int M, int N) {
  dim3 threads(32, 32);
  int grid_x = (N * 32 - 1) / 32;
  int grid_y = (M * 32 - 1) / 32;
  dim3 blocks(grid_x, grid_y);
  nn_real* out;
  // check_launch("pre cuda malloc ce gpu");
  cudaMalloc((void**)&out, sizeof(nn_real) * N);
  // check_launch("cuda_malloc_ce_gpu");
  
  cross_entropy_no_reduce_k<<<blocks, threads>>>(yc, y, M, N, out);
  // check_launch("cross_entropy_no_reduce_k");
  nn_real* loss;
  cudaMalloc((void**)&loss, sizeof(nn_real) * 1);
  // check_launch("cuda_malloc_ce_gpu_loss");
  sum(out, loss, N);
  // check_launch("ce_sum");
  nn_real* loss_cpu; 

  loss_cpu = (nn_real*) malloc(1 * sizeof(nn_real));
  cudaMemcpy(loss_cpu, loss, 1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  // check_launch("cuda_memcpy_loss");
  cudaFree(out);
  cudaFree(loss);

  nn_real mean_loss = *loss_cpu / N;
  free(loss_cpu);
  
  return mean_loss;
}

void gpu_norm(const nn_real* w, nn_real* out, int N) {
  dim3 threads(1024);
  int grid_x = (N + 1024 - 1) / 1024;
  dim3 blocks(grid_x);
  nn_real* w_square;
  cudaMalloc((void**)&w_square, sizeof(nn_real) * N);
  // check_launch("square");
  square_k<<<blocks, threads>>>(w, w_square, N);
  // check_launch("square_k");
  sum(w_square, out, N);
  cudaFree(w_square);
}


void sum(const nn_real* a, nn_real* out, int N) {
  int grid_x = (N * 1024 - 1) / 1024;
  reduce_sum<<<grid_x, 1024>>>(a, out, N);
  // check_launch("sum reduce");
}


//This is a huge place for optimization.
__global__
void transpose_kernel(const nn_real* a, nn_real* out, int M, int N) {
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < N && yIndex < M ){
      __shared__ nn_real t_shared[32][33];
      unsigned int index_in  = xIndex * M + yIndex;
      t_shared[threadIdx.y][threadIdx.x] = a[index_in];
      __syncthreads();
      unsigned int index_out = yIndex * N + xIndex;
      out[index_out] = t_shared[threadIdx.y][threadIdx.x]; 
   }
}


void transpose(const nn_real* a, nn_real* out, int M, int N){
  dim3 threads(32, 32);
  int grid_x = (N * 32 - 1) / 32;
  int grid_y = (M * 32 - 1) / 32;
  dim3 blocks(grid_x, grid_y);
  transpose_kernel<<<blocks, threads>>>(a, out, M, N);
}


__global__
void sigmoid_derivative_kernel(const nn_real* a, const nn_real* b, nn_real* out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    nn_real b_ = b[idx];
    out[idx] = a[idx] * b_ * (1 - b_);
  }
}

void sigmoid_derivative(const nn_real* activation, const nn_real* sig_result, nn_real* out, int N){
  int blocks = (N + 1024 - 1) / 1024;
  sigmoid_derivative_kernel<<<blocks, 1024>>>(activation, sig_result, out, N);
  // check_launch("Sigmoid kernel");
}


void rowwise_sum(const nn_real* a, nn_real* out, int M, int N) {
  int blocks = (M + 1024 - 1) / 1024;
  rowwise_sum_kernel<<<blocks, 1024>>>(a, out, M, N);

}


__global__
void scaled_difference_kernel(const nn_real* a, const nn_real* b, nn_real* c, int total, nn_real scalar) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total) {
    c[idx] = (a[idx] - b[idx]) * scalar;
  }
}


// returns (a - b) * 1 / N
void scaled_difference(const nn_real* a, const nn_real* b, nn_real* out, int M, int N){
  int grid_x = (N * 1024 - 1) / 1024;
  scaled_difference_kernel<<<grid_x, 1024>>>(a, b, out, M * N, ((nn_real) 1.0) / N);
  // check_launch("scaled difference");
}


__global__
void gradient_update_kernel(const nn_real* grad, nn_real* weights, nn_real scalar, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    weights[idx] = weights[idx] - scalar * grad[idx];
  }
}


void gradient_update(const nn_real* grad, nn_real* weights, nn_real scalar, int N) {
  // Note weights and grad have same dimension
  int grid_x = (N * 1024 - 1) / 1024;
  gradient_update_kernel<<<grid_x, 1024>>>(grad, weights, scalar, N);
}


__global__
void broadcast_rowwise_kernel(const nn_real* b, nn_real* out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M) {
    for(int i = 0; i < N; i++) {
      // out is an M x N matrix.
      out[idx + M * i] = b[idx];
    }
  }
}


// broad cast a column vector b of size M, N times in along axis 1.
void broadcast_rowwise(const nn_real* b, nn_real* out, int M, int N) {
    int grid_x = (N * 1024 - 1) / 1024;
    broadcast_rowwise_kernel<<<grid_x, 1024>>>(b, out, M, N);
}

size_t free_memory_on_gpu() {
  size_t t;
  size_t total;
  cudaMemGetInfo(&t, &total);
  return t;
}
