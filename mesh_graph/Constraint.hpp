#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP


/**
 * @brief Base Class of Constraint
 * 
 * Must implement operator
 * 
 */
class Constraint {


  /** Apply the constraint applying to @a n at time @a t.
   *
  */
    public:
      using GraphType = Graph<NodeData, EdgeData>;
      using NodeType  = typename GraphType::node_type;
      
      /**
       * @brief Base class, by default nodes are not fixed points
       * @post for all n in g, n._is_fixed_point = false
       * @param g Graph
       * @param t double timestep
       */
      virtual void operator()(GraphType& g, double t) {
        (void) t;
        for(auto it = g.node_begin(); it != g.node_end(); ++it) {
            NodeType node = *it;
            node.value().is_a_fixed_point = false;
        }
      }
};


#endif