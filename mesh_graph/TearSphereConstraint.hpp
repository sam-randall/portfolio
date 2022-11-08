#include "Constraint.hpp"
#ifndef TEARSPHERECONSTRAINT_HPP
#define TEARSPHERECONSTRAINT_HPP

/**
 * @brief 
 * 
 */
class TearSphereConstraint : public Constraint {

    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;

        /**
         * @brief operator function mutates the graph to make sure
         *  the constraint
         * is satisifed, which is if it's a sphere just remove that 
         * node.
         * 
         * @param g GraphType& graph
         * @param t double time
         */
        virtual void operator()(GraphType& g, double t) {
            (void) t;
            for(auto it = g.node_begin(); it != g.node_end(); ++it) {
                NodeType node = *it;
                Point center = Point(0.5, 0.5, -0.5);
                double radius = 0.15; 
                if (norm(node.position() - center) < radius ) {
                    // violated constraint
                    g.remove_node(node);
                }
            }
        }
};

#endif