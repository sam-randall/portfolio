#include "Constraint.hpp"
#ifndef SPHERECONSTRAINT_HPP
#define SPHERECONSTRAINT_HPP


struct sphere_if_necessary {
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    /**
     * @brief operator() that makes it a sphere functor
     * 
     * @param node 
     */
    void operator()(NodeType node) {
        Point center = Point(0.5, 0.5, -0.5);
        double radius = 0.15; 
        if (norm(node.position() - center) < radius ) {
            // violated constraint
            Point pos_2_c_vec = node.position() - center;
            double scalar = radius / norm(pos_2_c_vec);
            Point nearest = center + scalar * pos_2_c_vec;
            Point R_i = pos_2_c_vec / norm(pos_2_c_vec);
            node.position() =  nearest;
            double v_dot_R_i = dot(node.value().vel, R_i);
            node.value().vel -= v_dot_R_i * R_i;
        }
    }
};


/**
 * @brief 
 * 
 */
class SphereConstraint : public Constraint {

    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;

        /**
         * @brief Sphere Constraint
         * 
         * let sphere = Point center = Point(0.5, 0.5, -0.5);
                        double radius = 0.15; 
         * @post any node on sphere surface has vel = v dot R_i R_i
         * @post no node within sphere.
         * 
         * @param g GraphType
         * @param t double timestep
         */
        virtual void operator()(GraphType& g, double t) {
            (void) t;
            thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), sphere_if_necessary());
        }
};

#endif