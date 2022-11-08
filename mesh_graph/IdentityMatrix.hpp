/**
 * @file mtl_test.cpp
 * Implimentation file for interfacing with MTL4 and it's linear solvers.
 */

// HW3: Need to install/include Boost and MTL in Makefile
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

// HW3: YOUR CODE HERE
// Define a IdentityMatrix that interfaces with MTL

/**
 * @brief simple Identity Matrix that interfaces with MTL.
 * 
 */
class IdentityMatrix {
    /**
     * @brief 
     * 
     * Compute the product of a vecotr and the identity amtrix.
     * 
     */

    public:
        std::size_t n_cols;
        std::size_t n_rows;

        IdentityMatrix(std::size_t size) {
            this -> n_cols = size;
            this -> n_rows = size;
        }

    template <typename Vector>
    mtl::vec::mat_cvec_multiplier<IdentityMatrix, Vector>
    operator*(const Vector& v) const {
        return {*this, v};
    }

    /**
     * @brief Helper function to perform multiplication. Allows 
     * Assign::apply(a, b) resolves to an assignment operation
     * such as a += b, a -= b or a =b, 
     * @pre @a size(v) == size(w)
     */
    template<typename VectorIn, typename VectorOut, typename Assign>
    void mult(const VectorIn& v, VectorOut& w, Assign) const {
        Assign::apply(w, v);
    }

    private:
        
    
};


/**
 * @brief return number of rows in A.
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t num_rows(const IdentityMatrix& A) {
    return A.n_rows;
}

/**
 * @brief return number of columns in A
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t num_cols(const IdentityMatrix& A) {
    return A.n_cols;
}

/**
 * @brief return size (total number of elements) of A.
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t size(const IdentityMatrix& A) {
    return num_rows(A) * num_cols(A);
}

/** Traits that MTL uses to determine properties of our Identity Matrix */
namespace mtl {
    namespace ashape {

        /** Define IdentityMatrix to be a non scalar type. */
        template<>
        struct ashape_aux<IdentityMatrix> {
            typedef nonscal type;
        };
    }

    /** Identity Matrix implements the collection Concept 
     * with value_type and size_type */
    template<>
    struct Collection<IdentityMatrix> {
        typedef double value_type;
        typedef unsigned size_type;
    };
} 