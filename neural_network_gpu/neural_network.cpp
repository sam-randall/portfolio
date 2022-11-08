#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <unordered_map>
#include "cublas_v2.h"
#include "utils/gpu_util.h"
#include "gpu_func.h"
#include "mpi.h"

struct gpu_grads {
  std::vector<nn_real*> dW;
  std::vector<nn_real*> db;
};

struct gpu_cache {
  nn_real* X;
  std::vector<nn_real*> z;
  std::vector<nn_real*> a;
  nn_real* yc;
};

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

void norm_gpu_mat( const nn_real* mat, int M, int N) {
    // check_launch("begin norm gpu");
    nn_real* localW;
    localW = (nn_real*) malloc(sizeof(nn_real) * M * N);
    cudaMemcpy(localW, mat, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);
    // check_launch("norm_gpu_mat_memcpy");
    arma::Mat<nn_real> w = arma::Mat<nn_real>(localW, M, N);
    printf("%.18f\n", arma::norm(w));
    free(localW);

}

void print_gpu_mat( const nn_real* mat, int M, int N) {
    // check_launch("begin norm gpu");
    nn_real* localW;
    localW = (nn_real*) malloc(sizeof(nn_real) * M * N);
    cudaMemcpy(localW, mat, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);
    arma::Mat<nn_real> w = arma::Mat<nn_real>(localW, M, N);
    free(localW);
    w.print();
}

void flagpost(std::string s) {
  std::cout << s << std::endl;
}

nn_real parallel_norms(const NeuralNetwork& nn, const std::vector<nn_real*>& weights) {
  nn_real* sums;
  int sums_size = nn.num_layers * sizeof(nn_real);
  cudaMalloc((void**)&sums, nn.num_layers * sizeof(nn_real));


  for (int i = 0; i < nn.num_layers; ++i) {
    const nn_real* dw = weights[i];
    int w_elements = nn.W[i].n_cols * nn.W[i].n_rows;
    gpu_norm(dw, &sums[i], w_elements);
  }

  nn_real* sum_out;
  cudaMalloc((void**)&sum_out, sizeof(nn_real));

  nn_real* local_sum;
  local_sum = (nn_real*)malloc(1 * sizeof(nn_real));
  sum(sums, sum_out, nn.num_layers);

  cudaMemcpy(local_sum, sum_out, sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaFree(sums);
  cudaFree(sum_out);

  nn_real total_loss = *local_sum;
  free(local_sum);

  return total_loss;
}

nn_real norms(NeuralNetwork& nn) {
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1; //nn.W[0].n_rows x nn.X.n_cols
  sigmoid(z1, a1);
  cache.a[0] = a1; // cache.a[0] is the output of sigmoid.

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
  
}

void printShape(const arma::Mat<nn_real>& X) {
  std::cout << "Shape " << X.n_rows << "x" << X.n_cols << "\n";
}


// feedforward pass
void parallel_feedforward(NeuralNetwork& nn, const std::vector<nn_real*>& weights, const std::vector<nn_real*>& biases, nn_real* in, nn_real* layer1, nn_real* layer2, nn_real* sigmoid_out, nn_real* out,
                 struct gpu_cache& cache, int N) {

  cache.z.resize(2);
  cache.a.resize(2);

  nn_real beta = 1.0;
  nn_real alpha = 1.0;

  
  const nn_real* W0_ = weights[0];
  cache.X = in;

  broadcast_rowwise(biases[0], layer1, nn.W[0].n_rows, N);

  myGEMM(W0_, in, layer1, &alpha, &beta,
           nn.W[0].n_rows, N, nn.W[0].n_cols );

  cache.z[0] = layer1;

  int Z1_rows = nn.W[0].n_rows;
  
  sigmoid_gpu(layer1, sigmoid_out, Z1_rows * N);
  cache.a[0] = sigmoid_out;

  const nn_real* W1_ = weights[1];

  broadcast_rowwise(biases[1], layer2, nn.W[1].n_rows, N);

  size_t free_m = free_memory_on_gpu();

  myGEMM(W1_, sigmoid_out, layer2, &alpha, &beta,
           nn.W[1].n_rows,  N, nn.W[1].n_cols );
  cache.z[1] = layer2;
  
  softmax_gpu(layer2, out, nn.W[1].n_rows, N);

  cache.a[1] = cache.yc = out;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<nn_real>& y, nn_real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);

  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

void cpu_transpose(const nn_real* mat, nn_real* out, int M, int N) {

  nn_real* localW;
    localW = (nn_real*) malloc(sizeof(nn_real) * M * N);
    cudaMemcpy(localW, mat, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

    arma::Mat<nn_real> w = arma::Mat<nn_real>(localW, M, N);
    arma::Mat<nn_real> transpose = w.t();
    nn_real* t_ = transpose.memptr();
    cudaMemcpy(out, t_, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
    free(localW);

}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void parallel_backprop(const NeuralNetwork& nn, const std::vector<nn_real*> &weights, const std::vector<nn_real*> &biases, const nn_real* y, nn_real reg,
              const struct gpu_cache& bpcache, struct gpu_grads& bpgrads, int N, std::unordered_map<string, nn_real*>& intermediate_ptrs) {

  if (N == 0) {
    throw;
  }

  nn_real* difference = intermediate_ptrs["difference"];
  nn_real* a0_t = intermediate_ptrs["a0_transpose"];
  nn_real* w1_t = intermediate_ptrs["w1_transpose"];
  nn_real* d_deriv_a1 = intermediate_ptrs["a1_derivative"];
  nn_real* sigmoid_deriv = intermediate_ptrs["sigmoid_derivative"];
  nn_real* xt = intermediate_ptrs["x_transpose"];
  scaled_difference(bpcache.yc, y, difference, nn.H[2], N);
  // bpcache.a[0] has shape = (difference.n_cls x nn.W[1].n_cols) = 
  
  transpose(bpcache.a[0], a0_t, nn.W[1].n_cols, N);

  nn_real a = 1.0;
  
  myGEMM(difference, a0_t, weights[1], bpgrads.dW[1], &a, &reg, nn.W[1].n_rows, nn.W[1].n_cols, N);
  // Writes to bpgrads.db[1].
  rowwise_sum(difference, bpgrads.db[1], nn.H[2], N);
  transpose(weights[1], w1_t, nn.W[1].n_rows, nn.W[1].n_cols);
  myGEMMOverwrite(w1_t, difference, d_deriv_a1, &a, nn.W[1].n_cols, N, nn.H[2]);
  // sigmoid_deriv has shape (nn.W[1].n_cols x N)
  sigmoid_derivative(d_deriv_a1, bpcache.a[0], sigmoid_deriv, nn.W[1].n_cols * N);
  transpose(bpcache.X, xt, nn.W[0].n_cols, N);
  myGEMM(sigmoid_deriv, xt, weights[0], bpgrads.dW[0], &a, &reg, nn.W[1].n_cols, nn.W[0].n_cols, N);
  rowwise_sum(sigmoid_deriv, bpgrads.db[0], nn.W[1].n_cols,  N);


}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork& nn, const arma::Mat<nn_real>& yc,
             const arma::Mat<nn_real>& y, nn_real reg) {
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));
  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  return loss;
}


/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
             arma::Row<nn_real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
           const arma::Mat<nn_real>& y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);


      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
          
        }
        // std::cout << "Loss at iteration " << iter << " of epoch " << epoch
        //           << "/" << epochs << " = "
        //           << loss(nn, bpcache.yc, y_batch, reg) << "\n";

      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        if (print_every > 0 && iter % print_every == 0) {
          printf("Iteration %i: CPU Train dW[%i]=%.18f\n", iter, i, arma::norm(bpgrads.dW[i]));
        }
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        // if (print_every > 0 && iter % print_every == 0) {
        //   printf("Train db[%i]: %.18f\n", i, arma::norm(bpgrads.db[i]));
        // }
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        save_cpu_data(nn, iter);
      }

      iter++;
    }
  }
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                    const arma::Mat<nn_real>& y, nn_real learning_rate,
                    std::ofstream& error_file, 
                    nn_real reg, const int epochs, const int batch_size,
                    int print_every, int debug) {
  int rank, num_procs;
  int root = 0;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */

  // TODO


  std::vector<nn_real*> weights;
  std::vector<nn_real*> biases;

  for(int i = 0; i < nn.num_layers;i++) {
    nn_real* dwptr; 
    nn_real* dbptr; 
    
    nn_real* hwptr = nn.W[i].memptr();
    nn_real* hbptr = nn.b[i].memptr();

    cudaMalloc((void**)&dwptr, sizeof(nn_real) * nn.H[i + 1] * nn.H[i]);
    cudaMemcpy(dwptr, hwptr, sizeof(nn_real) * nn.H[i + 1] * nn.H[i], cudaMemcpyHostToDevice);

    weights.push_back(dwptr);

    cudaMalloc((void**)&dbptr, sizeof(nn_real) * nn.H[i + 1]);
    cudaMemcpy(dbptr, hbptr, sizeof(nn_real) * nn.H[i + 1], cudaMemcpyHostToDevice);
    biases.push_back(dbptr);
  }


  nn_real* x_in;
  cudaMalloc((void**)&x_in, sizeof(nn_real) * batch_size * nn.H[0]);

  nn_real* y_in;
  cudaMalloc((void**)&y_in, sizeof(nn_real) * batch_size * nn.H[2]);

  nn_real* l_1_out;
  cudaMalloc((void**)&l_1_out, sizeof(nn_real) * batch_size * nn.H[1]);

  nn_real* sigmoid_out;
  cudaMalloc((void**)&sigmoid_out, sizeof(nn_real) * batch_size * nn.H[1]);

  nn_real* l_2_out;
  cudaMalloc((void**)&l_2_out, sizeof(nn_real) * batch_size * nn.H[2]);

  nn_real* out;
  cudaMalloc((void**)&out, sizeof(nn_real) * batch_size * nn.H[2]);

  struct gpu_cache bpcache;
  struct gpu_grads bpgrads;
  struct grads hgrads;

  for(int i = 0; i < nn.num_layers; i++) {
    nn_real* ptr;
    cudaMalloc((void**)&ptr, nn.H[i] * nn.H[i + 1] * sizeof(nn_real));

    nn_real* b_ptr;
    cudaMalloc((void**)&b_ptr, nn.H[i + 1] * sizeof(nn_real));
  
    bpgrads.dW.push_back(ptr);
    bpgrads.db.push_back(b_ptr);
  }


  std::unordered_map<string, nn_real*> backprop_d_ptrs;

  nn_real* difference;
  nn_real* a0_t;
  nn_real* w1_t;
  nn_real* d_deriv_a1;
  nn_real* sigmoid_deriv;
  nn_real* xt;


  cudaMalloc((void**)&difference, N * sizeof(nn_real) * nn.H[2]);
  cudaMalloc((void**)&a0_t, N * nn.W[1].n_cols * sizeof(nn_real));
  cudaMalloc((void**) &w1_t, nn.W[1].n_rows * nn.W[1].n_cols * sizeof(nn_real));
  cudaMalloc((void**) &d_deriv_a1, sizeof(nn_real) * nn.W[1].n_cols * N);
  cudaMalloc((void**)&sigmoid_deriv, nn.W[1].n_cols * N * sizeof(nn_real)); // save size as da1
  cudaMalloc((void**)&xt, nn.W[0].n_cols * N * sizeof(nn_real));

  backprop_d_ptrs["difference"] = difference;
  backprop_d_ptrs["a0_transpose"] = a0_t;
  backprop_d_ptrs["w1_transpose"] = w1_t;
  backprop_d_ptrs["a1_derivative"] = d_deriv_a1;
  backprop_d_ptrs["sigmoid_derivative"] = sigmoid_deriv;
  backprop_d_ptrs["x_transpose"] = xt;

  int batches_per_mega_batch = 8;
  int num_batches = (N + batch_size - 1) / batch_size;

  
  /* allocate memory before the iterations */
  // Data sets
  
  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_mega_batches = num_batches / 8;
    for (int batch = 0; batch < num_batches; ++batch) {

      

        int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
        int x_n_channels = nn.W[0].n_cols;
        int this_batch_size = (last_col - batch * batch_size) + 1; // inclusive.

        if (this_batch_size % num_procs != 0) {
          throw;
        }

        int images_to_send = this_batch_size / num_procs;
        int item_count = images_to_send * x_n_channels;
        int y_item_count = images_to_send * nn.H[2];

        nn_real* tempbuf = new nn_real[item_count];;
        nn_real* ybuf = new nn_real[y_item_count];

        if( rank == root) {

            arma::Mat<nn_real> X_main_batch = X.cols(batch * batch_size, last_col);
            arma::Mat<nn_real> y_main_batch = y.cols(batch * batch_size, last_col);
            nn_real* x_ptr_ = X_main_batch.memptr();
            nn_real* y_ptr_ = y_main_batch.memptr();

            int res = MPI_Scatter(x_ptr_, item_count, MPI_FP, tempbuf, item_count, MPI_FP, root, MPI_COMM_WORLD);
            int res_y = MPI_Scatter(y_ptr_, y_item_count, MPI_FP, ybuf, y_item_count, MPI_FP, root, MPI_COMM_WORLD);

        } else {

          int res = MPI_Scatter(NULL, item_count, MPI_FP, tempbuf, item_count, MPI_FP, root, MPI_COMM_WORLD);
          int res_y = MPI_Scatter(NULL, item_count, MPI_FP, ybuf, y_item_count, MPI_FP, root, MPI_COMM_WORLD);

        }

        arma::Mat<nn_real> X_batch(tempbuf, x_n_channels, images_to_send);
        arma::Mat<nn_real> y_batch(ybuf, nn.H[2], images_to_send);
        int n_cols_in_batch = X_batch.n_cols;

        const nn_real* X_ = X_batch.memptr();
         cudaMemcpy(x_in, X_, X_batch.n_rows * n_cols_in_batch * sizeof(nn_real), cudaMemcpyHostToDevice);

        parallel_feedforward(nn, weights, biases,  x_in, l_1_out, l_2_out, sigmoid_out, out, bpcache, n_cols_in_batch);

        nn_real* y_ptr = y_batch.memptr();
        cudaMemcpy(y_in, y_ptr, n_cols_in_batch * nn.H[2] * sizeof(nn_real), cudaMemcpyHostToDevice);
        parallel_backprop(nn, weights, biases, y_in, reg, bpcache, bpgrads, n_cols_in_batch, backprop_d_ptrs);

        if (print_every <= 0) {
          print_flag = batch == 0;
        } else {
          print_flag = iter % print_every == 0;
        }
        

        for(int i = 0; i < nn.num_layers; i++) {
            int data_size_w = nn.H[i + 1] * nn.H[i] * sizeof(nn_real);
            nn_real* global_dw = new nn_real[data_size_w]; // this only puts it on CPU.
            nn_real* local_grad = new nn_real[data_size_w];
            cudaMemcpy(local_grad, bpgrads.dW[i], data_size_w, cudaMemcpyDeviceToHost);
            MPI_Allreduce(local_grad, global_dw, nn.H[i + 1] * nn.H[i], MPI_FP, MPI_SUM, MPI_COMM_WORLD);
            delete [] local_grad;

            // Need to store global_dw
            cudaMemcpy(bpgrads.dW[i], global_dw, data_size_w, cudaMemcpyHostToDevice);
            

            int data_size_b = nn.H[i + 1] * sizeof(nn_real);
            nn_real* global_db = new nn_real[data_size_b]; // this only puts it on CPU.
            nn_real* local_grad_db = new nn_real[data_size_b];
            cudaMemcpy(local_grad_db, bpgrads.db[i], data_size_b, cudaMemcpyDeviceToHost);
            MPI_Allreduce(local_grad_db, global_db, nn.H[i + 1], MPI_FP, MPI_SUM, MPI_COMM_WORLD);

            delete [] local_grad_db;

            cudaMemcpy(bpgrads.db[i], global_db, data_size_b, cudaMemcpyHostToDevice);

            delete [] global_dw;
            delete [] global_db;
        }



        // Do a local update of all weights and biases.

        for (int i = 0; i < nn.num_layers; i++) {
          gradient_update(bpgrads.dW[i], weights[i], learning_rate / num_procs, nn.W[i].n_rows * nn.W[i].n_cols);
          gradient_update(bpgrads.db[i], biases[i], learning_rate / num_procs, nn.H[i + 1]);
        }

        delete [] tempbuf;
        delete [] ybuf;

        

        if (debug && (rank == 0) && print_flag){
            // Copy GPU to local.
          for(int i = 0; i < nn.num_layers; i++) {
              int n_w_elements = nn.W[i].n_cols * nn.W[i].n_rows;
              int n_b_elements = nn.H[i + 1];
              int float_size = sizeof(nn_real);
              nn_real* raw_w = nn.W[i].memptr();
              nn_real* raw_b = nn.b[i].memptr();
              cudaMemcpy(raw_w, weights[i], float_size * n_w_elements, cudaMemcpyDeviceToHost);
              cudaMemcpy(raw_b, biases[i], float_size * n_b_elements, cudaMemcpyDeviceToHost);
              nn.W[i] = arma::Mat<nn_real>(raw_w, nn.W[i].n_rows, nn.W[i].n_cols);
              nn.b[i] = arma::Col<nn_real>(raw_b, nn.H[i + 1]);


          }
            save_gpu_error(nn, iter, error_file);            
        }

        iter++;

    }
  }

  // Copy GPU to local.
  for(int i = 0; i < nn.num_layers; i++) {
            int n_w_elements = nn.W[i].n_cols * nn.W[i].n_rows;
            int n_b_elements = nn.H[i + 1];
            int float_size = sizeof(nn_real);
            nn_real* raw_w = nn.W[i].memptr();
            nn_real* raw_b = nn.b[i].memptr();
            cudaMemcpy(raw_w, weights[i], float_size * n_w_elements, cudaMemcpyDeviceToHost);
            cudaMemcpy(raw_b, biases[i], float_size * n_b_elements, cudaMemcpyDeviceToHost);
            nn.W[i] = arma::Mat<nn_real>(raw_w, nn.W[i].n_rows, nn.W[i].n_cols);
            nn.b[i] = arma::Col<nn_real>(raw_b, nn.H[i + 1]);


  }

  for(unsigned int i = 0; i < weights.size() ;i++) {
    nn_real* dwptr = weights[i];
    nn_real* dbptr = biases[i];

    cudaFree(dwptr);
    cudaFree(dbptr);

    cudaFree(bpgrads.dW[i]);
    cudaFree(bpgrads.db[i]);
  }

  cudaFree(x_in);
  cudaFree(y_in);
  cudaFree(l_1_out);
  cudaFree(l_2_out);
  cudaFree(out);


  for(auto id : backprop_d_ptrs) {
    cudaFree(id.second);
  }


}
