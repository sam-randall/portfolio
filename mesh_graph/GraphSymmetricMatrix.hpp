/**
 * @file GraphSymmetricMatrix.hpp
 * Implimentation file for treating the Graph as a MTL Matrix
 */

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

#include <boost/numeric/itl/iteration/cyclic_iteration.hpp>
#include <algorithm>
#include <fstream>

#include "CME212/Color.hpp"
#include "CME212/Util.hpp"
#include "CME212/Point.hpp"
#include "CME212/BoundingBox.hpp"

#include "Graph.hpp"

/** Custom structure of data to store with Nodes */
struct NodeBoundaryData {
  char boundary_val;
  /**
   * @brief Construct a new Node Boundary Data object
   * 
   */
  NodeBoundaryData() : boundary_val(0) {}

};

// Define a GraphSymmetricMatrix that maps
// your Graph concept to MTL's Matrix concept. This shouldn't need to copy or
// modify Graph at all!
using GraphType = Graph<NodeBoundaryData,char>;  //<  DUMMY Placeholder

using NodeType  = typename GraphType::node_type;

using EdgeType  = typename GraphType::edge_type;

/**
 * @brief Graph Symmetric Matrix takes in a graph and interprets it as a matrix.
 * 
 */
class GraphSymmetricMatrix {
  public: 
    GraphSymmetricMatrix(const GraphType* g) {
      g_ = g;
    }

    template <typename Vector>
    mtl::vec::mat_cvec_multiplier<GraphSymmetricMatrix, Vector>
    operator*(const Vector& v) const {
        return {*this, v};
    }

    std::size_t get_dim() const {
      return g_ -> num_nodes();
    }

    /**
     * @brief Helper function to perform multiplication. Allows 
     * Assign::apply(a, b) resolves to an assignment operation
     * such as a += b, a -= b or a =b, 
     * @pre @a size(v) == size(w)
     */
    template<typename VectorIn, typename VectorOut, typename Assign>
    
    void mult(const VectorIn& v, VectorOut& w, Assign) const {

        for(auto it = g_ -> node_begin(); it != g_ -> node_end(); ++it) {
          NodeType n = *it;
          unsigned int i = n.index();
          char on_boundary = n.value().boundary_val;

          if (on_boundary) {
            Assign::apply(w[i], v[i]);
          } else {

            double deg = (double) n.degree();
            double running_sum = -deg * v[i];
            for (auto adj_edge_it = n.edge_begin();
             adj_edge_it != n.edge_end();
             ++adj_edge_it) {

               EdgeType e = *adj_edge_it;
               auto dst = e.node2();
               char on_boundary_adj_val = dst.value().boundary_val;

               bool is_adjacent_node_not_on_boundary = (on_boundary_adj_val == 0);
              
              // if it's NOT on the boundary.
               if (is_adjacent_node_not_on_boundary) {
                running_sum += v[dst.index()];
               }
               
            }
            Assign::apply(w[i], running_sum);
            
          }
        }
    }

  private:
    const GraphType* g_;
};

/** Remove all the nodes in graph @a g whose posiiton is within Box3D @a bb.
 * @param[in,out] g  The Graph to remove nodes from
 * @param[in]    bb  The BoundingBox, all nodes inside this box will be removed
 * @post For all i, 0 <= i < @a g.num_nodes(),
 *        not bb.contains(g.node(i).position())
 */
void remove_box(GraphType& g, const Box3D& bb) {

  for(auto it = g.node_begin(); it != g.node_end(); ++it) {
      NodeType node = *it;

      // while loop because it could plcae a valid node in this same spot!
      while (bb.contains(node.position()) ) {
          // violated constraint
          g.remove_node(node);
      } 
  }
  return;
}

/**
 * @brief return number of rows in A
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t num_rows(const GraphSymmetricMatrix& A) {
    return A.get_dim();
}

/**
 * @brief return number of columns in A.
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t num_cols(const GraphSymmetricMatrix& A) {
    return A.get_dim();
}

/**
 * @brief returns number of elements in A
 * 
 * @param A 
 * @return std::size_t 
 */
inline std::size_t size(const GraphSymmetricMatrix& A) {
    return num_rows(A) * num_cols(A);
}

/** Traits that MTL uses to determine properties of our GraphSymmetrixMatrix */
namespace mtl {
    namespace ashape {

        /** Define GraphSymmetrixMatrix to be a non scalar type. */
        template<>
        struct ashape_aux<GraphSymmetricMatrix> {
            typedef nonscal type;
        };
    }

    /** Identity GraphSymmetrixMatrix implements the collection Concept 
     * with value_type and size_type */
    template<>
    struct Collection<GraphSymmetricMatrix> {
        typedef double value_type;
        typedef unsigned size_type;
    };
} 

typedef mtl::dense_vector<double> vec_d;

/**
 * @brief NodeColor colors a node based on the current node solution.
 * 
 */
struct NodeColor {

  /**
   * @brief Construct a new Node Color object
   * 
   * @param vec 
   */
  NodeColor(vec_d& vec) : solution_(vec) {
    min_ = mtl::min(vec);
    max_ = mtl::max(vec);
  }

  /**
   * @brief operator takes in a Node object and returns a Color.
   * 
   * @param n 
   * @return CME212::Color 
   */
  CME212::Color operator()(NodeType& n) {

    double diff = max_ - min_;
    if (diff < 0) {
      diff = 1;
    }
    double node_sol = solution_[n.index()];
    double heat_val = (node_sol - min_) / (diff);
    float fl_color = (float) heat_val;
    float v = fl_color <= 1 ? fl_color : 1;
    float non_neg = v >= 0 ? v : 0;
    CME212::Color color = CME212::Color::make_heat(non_neg);
    return color;
  }

  private:
    /** Solution vector returned to us from some Solver, e.g CGSolver */
    const vec_d& solution_;

    double min_ = 0;
    double max_ = 0;
    
};

/**
 * @brief 
 * 
 */
struct NodePosition {

  /**
   * @brief Construct a new Node Position object with a solution vector
   * which can mutate.
   * 
   * @param vec 
   */
  NodePosition(const vec_d& vec) : solution_(vec) {}

  /**
   * @brief gets a position from a node using the solution for the z axis.
   * 
   * @param n 
   * @return Point 
   */
  Point operator()(NodeType& n) {
      auto idx = n.index();
      auto pos = n.position();
      return Point(pos.x, pos.y, solution_[idx]);
  }

  private:
    /** Solution vector returned to us from some Solver, e.g CGSolver */
    const vec_d& solution_;


};
