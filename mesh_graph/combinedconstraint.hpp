#include "Constraint.hpp"

#ifndef COMBINEDCONSTRAINT_HPP
#define COMBINEDCONSTRAINT_HPP


/**
 * @brief Combined Constraint implements a combination constraint
 * by summing up the constraints passed to it.
 * 
 */
class CombinedConstraint : public Constraint {
    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;
        /**
         * @brief operator() returns sum of constraints.
         * 
         * @param g Graph
         * @param t double timestep
         */
        virtual void operator()(GraphType& g, double t) {
            for(Constraint* constraint_ptr : constraints_) {
                auto &constraint = *constraint_ptr;
                constraint(g, t);
            }
        }
        /**
         * @brief Construct a new Combined Constraint object
         * 
         * @param constraints std::vector<Constraint*>
         */
        CombinedConstraint(std::vector<Constraint*> constraints) {
            constraints_ = constraints;
        }

    private:
        // where the constraints live.
        std::vector<Constraint*> constraints_;
};

/**
 * @brief make_combined_constraint combines3 constraints into one
 * accepts any child class of Constraint
 * accepts 2 or 3 arguments.
 * @param c1 constraint
 * @param c2 constraint
 * @param c3 constraint - default value is Constraint() base class
 * @return CombinedConstraint 
 */
CombinedConstraint make_combined_constraint(Constraint&& c1, Constraint&& c2, Constraint&& c3 = Constraint()) {
    std::vector<Constraint*> constraints;

    constraints.push_back(&c1);
    constraints.push_back(&c2);
    constraints.push_back(&c3);

    return CombinedConstraint(constraints);
}

#endif