#include "CME212/Point.hpp"
#include "Graph.hpp"
#include "Data.hpp"

#ifndef FORCE_HPP
#define FORCE_HPP

/**
 * @brief Base Class 
 * 
 * child classes must implement operator
 * force(n, t)
 * 
 */
class Force {

  /** Return the force applying to @a n at time @a t.
   *
   * For HW2 #1, this is a combination of mass-spring force and gravity,
   * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
   * model that by returning a zero-valued force. */
    public:
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
      virtual Point operator()(NodeType n, double t) const {
        (void) n;
        (void) t;
        return Point(0);  
      }
};

#endif