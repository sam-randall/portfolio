/**
 * @file poisson.cpp
 * Test script for treating for using the GraphSymmetricMatrix class
 * and solving a Poisson equation.
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles.
 * Second file: Eges (one per line) defined by 2 indices into the point list
 *              of the first file.
 *
 * Launches an SFML_Viewer to visualize the solution.
 */

#include "CME212/SFML_Viewer.hpp"
#include "GraphSymmetricMatrix.hpp"

#include <cmath>

// HW3: YOUR CODE HERE
// Define visual_iteration that inherits from cyclic_iteration

namespace itl {

  /**
   * @brief visual_iteration inherits from cyclic iteration 
   * and implements a visualization of the solution a solver
   * progresses
   * 
   * @tparam Real 
   * @tparam OStream 
   */
  template <class Real, class OStream = std::ostream>
  class visual_iteration : public cyclic_iteration<Real> {

    /** super allows us to easily call methods in cyclic_iteration */
    typedef cyclic_iteration<Real> super;

    /** visual_iteration self methods */
    typedef visual_iteration self;

    /** Printing residual : identical to cyclic_iteration */
    void print_resid(){
      
      if (!this->my_quite && this->i % super::cycle == 0) {
        if (multi_print || this->i != super::last_print) { // Avoid multiple print-outs in same iteration
          out << "iteration " << this->i << ": resid " << this->resid() << std::endl;
          super::last_print= this->i;
        }
      }
    }

    public:

      template <class Vector>
      visual_iteration(const Vector& r0, int max_iter_, Real tol_, Real atol_ = Real(0), int cycle_ = 100,
                      OStream& out = std::cout)
      : super(r0, max_iter_, tol_, atol_, cycle_, out), out(out) {}

      /**
       * @brief Construct a new visual iteration object
       * 
       * @tparam Vector 
       * @param r0 
       * @param max_iter_ 
       * @param tol_ 
       * @param viewer 
       * @param graph 
       * @param u 
       * @param atol_ 
       * @param cycle_ 
       * @param out 
       */
      template <class Vector>
      visual_iteration(const Vector& r0, int max_iter_, Real tol_, CME212::SFML_Viewer* viewer, GraphType* graph, vec_d* u, Real atol_ = Real(0), int cycle_ = 100, 
                      OStream& out = std::cout)
      : visual_iteration(r0, max_iter_, tol_, atol_, cycle_, out) {
        viewer_ = viewer;
        graph_ = graph;
        node_map_ = viewer_ -> empty_node_map(*graph_);
        solution_ = u;
      }
      

      /**
       * @brief finished()
       * 
       * @return true 
       * @return false 
       */
      bool finished() { return super::finished(); }


      /**
       * @brief finished.
       * 
       * @tparam T 
       * @param r 
       * @return true 
       * @return false 
       */
      template <typename T>
      bool finished(const T& r){
          bool ret= super::finished(r);
          print_resid();

          // Visualization: only change to this class.
          update_viewer();
          return ret;
      }

      /**
       * @brief update viewer by clearing 
       * setting label, adding nodes and edges with corresponding coloring 
       * position calculation.
       * 
       */
      void update_viewer() {
        NodeColor colorMap(*solution_);
        NodePosition node2Position(*solution_);
        viewer_ -> clear();
        node_map_.clear();
        viewer_ -> set_label(this -> i);
        viewer_ -> add_nodes(graph_ -> node_begin(), graph_ -> node_end(), colorMap, node2Position, node_map_);
        viewer_ -> add_edges(graph_ -> edge_begin(), graph_ ->edge_end(), node_map_);
        viewer_ -> center_view();
      }

      /** same as super method. */
      inline self& operator++() { ++this->i; return *this; }

      /** same as super method*/
      inline self& operator+=(int n) { this->i+= n; return *this; }

      /** same as super method. */
      operator int() const { return error_code(); }

      /** just call super. */
      bool is_multi_print() const { return super::is_multi_print(); }

      /** set_multi_print calls super. */
      void set_multi_print(bool m) { 
        (void) m;
        super::set_multi_print(); }

      /** error_code calls super method */
      int error_code() const { return super::error_code(); }

    protected:
      int        cycle, last_print;
      bool       multi_print;
      OStream&   out;

      /** viewer_ passed as pointer, updated. */
      CME212::SFML_Viewer* viewer_;

      /** graph nodes accessed */
      GraphType* graph_;

      /** solution_ is passed by reference and updated by Solver and visualized here */
      vec_d* solution_;

      /** node_map property of viewer. this seems like unideal set up, but so be it.*/
      std::map<NodeType, unsigned> node_map_;
  };

} // namespace itl


bool position_is_inf_norm_near_other_point_under_thresh(const Point& position, const Point& other, double threshold ) {
  Point difference = position - other;
  double norm = norm_inf(difference);
  return norm < threshold;
}

/**
 * @brief evaluate g(x) given case 1, 2, 3.
 * 
 * @param val 
 * @return double 
 */
double evaluate_boundary_val(char val) {
  switch(val) {
    case 1:
      return 0.0;
      break;
    case 2:
      return -0.2;
      break;
    case 3:
      return 1.0;
      break;
    default:
      throw;
      return -1;
  }
}


vec_d initialize_b(GraphType& graph, double h) {
  vec_d b(graph.num_nodes()); 
  for(unsigned int i = 0; i < graph.num_nodes(); ++i) {
    NodeType n = graph.node(i);

    char boundary_val = n.value().boundary_val;

    if (boundary_val) { // not equal to 0.
      b[i] = evaluate_boundary_val(boundary_val);
    } else {
      Point p = n.position();
      double norm = norm_1(p);

      double fx_i = 5 * std::cos(norm);

      double g_sum = 0;

      for(auto it = n.edge_begin(); it != n.edge_end(); ++it) {
        EdgeType e = *it;
        NodeType adjacent_node = e.node2();
        char boundary_val = adjacent_node.value().boundary_val;
      
        if (boundary_val) { 
          g_sum += evaluate_boundary_val(boundary_val);
        }
      }

      b[i] = (h * h * fx_i) - g_sum;
    }
  }
  return b;
}


int main(int argc, char** argv)
{
  // Check arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " NODES_FILE TETS_FILE\n";
    exit(1);
  }

  // Define an empty Graph
  GraphType graph;

  {
    // Create a nodes_file from the first input argument
    std::ifstream nodes_file(argv[1]);
    // Interpret each line of the nodes_file as a 3D Point and add to the Graph
    std::vector<NodeType> node_vec;
    Point p;
    while (CME212::getline_parsed(nodes_file, p))
      node_vec.push_back(graph.add_node(2*p - Point(1,1,0)));

    // Create a tets_file from the second input argument
    std::ifstream tets_file(argv[2]);
    // Interpret each line of the tets_file as four ints which refer to nodes
    std::array<int,4> t;
    while (CME212::getline_parsed(tets_file, t)) {
      graph.add_edge(node_vec[t[0]], node_vec[t[1]]);
      graph.add_edge(node_vec[t[0]], node_vec[t[2]]);
      graph.add_edge(node_vec[t[1]], node_vec[t[3]]);
      graph.add_edge(node_vec[t[2]], node_vec[t[3]]);
    }
  }

  // Get the edge length, should be the same for each edge
  auto it = graph.edge_begin();
  assert(it != graph.edge_end());
  double h = norm((*it).node1().position() - (*it).node2().position());


  // Make holes in our Graph
  remove_box(graph, Box3D(Point(-0.8+h,-0.8+h,-1), Point(-0.4-h,-0.4-h,1)));
  remove_box(graph, Box3D(Point( 0.4+h,-0.8+h,-1), Point( 0.8-h,-0.4-h,1)));
  remove_box(graph, Box3D(Point(-0.8+h, 0.4+h,-1), Point(-0.4-h, 0.8-h,1)));
  remove_box(graph, Box3D(Point( 0.4+h, 0.4+h,-1), Point( 0.8-h, 0.8-h,1)));
  remove_box(graph, Box3D(Point(-0.6+h,-0.2+h,-1), Point( 0.6-h, 0.2-h,1)));

  // HW3: YOUR CODE HERE
  // Define b using the graph, f, and g.
  // Construct the GraphSymmetricMatrix A using the graph
  // Solve Au = b using MTL.

  for (auto it = graph.node_begin(); it  != graph.node_end(); ++it) {
    NodeType node = *it;

    auto position = node.position();
    auto bb = Box3D(Point(-0.6, -0.2, -1), Point( 0.6, 0.2, 1));
    bool in_bounding_box = bb.contains(node.position());


    if (std::abs(norm_inf(position) - 1) < 1e-9){
      node.value().boundary_val = 1;
    } else if (position_is_inf_norm_near_other_point_under_thresh(position, Point(0.6, 0.6, 0), 0.2)) {
      node.value().boundary_val = 2;
    } else if (position_is_inf_norm_near_other_point_under_thresh(position, Point(0.6, -0.6, 0), 0.2)) {
      node.value().boundary_val = 2;
    } else if (position_is_inf_norm_near_other_point_under_thresh(position, Point(-0.6, 0.6, 0), 0.2)) {
      node.value().boundary_val = 2;
    } else if (position_is_inf_norm_near_other_point_under_thresh(position, Point(-0.6, -0.6, 0), 0.2)) {
      node.value().boundary_val = 2;
    } else if (in_bounding_box) {
      node.value().boundary_val = 3;
    } else {
      node.value().boundary_val = 0; // to be explicit.
    }
    
  }

  GraphSymmetricMatrix mat(&graph);

  
  vec_d x(graph.num_nodes());

  vec_d b = initialize_b(graph, h);

  CME212::SFML_Viewer viewer;

  auto sim_thread = std::thread([&](){
      
    itl::visual_iteration<double> iter(b, 1000, 10.e-10, &viewer, &graph, &x, 0, 50);
    itl::pc::identity<GraphSymmetricMatrix> precond(mat);

    itl::cg(mat, x, b, precond, iter);
  });  // simulation thread

  viewer.event_loop();

  return 0;
}
