/**
 * @file mtl_test.cpp
 * Test script for interfacing with MTL4 and it's linear solvers.
 */

#include "IdentityMatrix.hpp"
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

int main(){

  typedef mtl::dense_vector<double> vec_d;
  vec_d b(20);

  iota(b);

  IdentityMatrix A(size(b));

  vec_d x(20);
  x = b - 1;


  std::cout << b << std::endl;
  std::cout << x << std::endl;
  assert(x != b);
  
  itl::noisy_iteration<double> iter(b, 500, 1.e-6);
  itl::pc::identity<IdentityMatrix> precond(A);
  itl::cg(A, x, b, precond, iter);


  assert(x == b);
  // HW3: YOUR CODE HERE
  // Construct an IdentityMatrix and "solve" Ix = b
  // using MTL's conjugate gradient solver

  return 0;
}
