#include "Constraint.hpp"

#ifndef PINCONSTRAINT_HPP
#define PINCONSTRAINT_HPP

/**
 * @brief pin if necessary constructor
 * 
 */
struct pin_if_necessary {
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;

    void operator()(NodeType node) {
        if (node.position() == Point(0)) {
            node.value().is_a_fixed_point = true;
            node.value().vel = Point(0);
        } else if (node.position() == Point(1,0,0)) {
            node.value().is_a_fixed_point = true;
            node.value().vel = Point(0);
        }
    }
};



/**
 * @brief PinConstraint pins 0, 0, 0 and 1, 0, 0 nodes.
 * 
 */
class PinConstraint : public Constraint {

  /** Return the force applying to @a n at time @a t.
   *
   * For HW2 #1, this is a combination of mass-spring force and gravity,
   * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
   * model that by returning a zero-valued force. */
    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;
        /**
         * @brief operator which implements the above PinConstraint.
         * called using constrain(g, t);
         * @post for all nodes
         * @post if node.position() == Point(0), then node.vel = Point(0)
         * @post if node.position() == Point(1, 0, 0), then node.vel = Point(0)
         * @param g GraphType& g;
         * @param t double t;
         */
        virtual void operator()(GraphType& g, double t) {
            (void) t;

            thrust::for_each(thrust::omp::par,
             g.node_begin(),
              g.node_end(),
               pin_if_necessary());
            
        }
};


#endif


