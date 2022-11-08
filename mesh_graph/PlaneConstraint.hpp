#include "Constraint.hpp"
#ifndef PLANECONSTRAINT_HPP
#define PLANECONSTRAINT_HPP

/**
 * @brief Plane Constraint that raises things as needed.
 * 
 */
struct raise_if_necessary {
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    void operator()(NodeType node) {
        if (!node.value().is_a_fixed_point) {
            if (node.position().z < -0.75) {
                node.value().vel.z = 0;
                node.position().z = -0.75;
            } 
        } 
    }
};

class PlaneConstraint : public Constraint {

  /** Return the force applying to @a n at time @a t.
   *
   * For HW2 #1, this is a combination of mass-spring force and gravity,
   * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
   * model that by returning a zero-valued force. */
    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;
        /**
         * @brief operator takes in constraint(g, t)
         * 
         * @post Graph g obeys constraint
         * all nodes >= -0.75
         * all nodes.position.z == -0.75
         * 
         * @param g GraphType& graph to mutate.
         * @param t double timestep
         */
        virtual void operator()(GraphType& g, double t) {
            (void) t;
            thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), raise_if_necessary());
        }
        
};



#endif