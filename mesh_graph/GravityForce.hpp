
#include "Force.hpp"


/**
 * @brief Implements force due to gravity.
 * 
 */
class GravityForce : public Force {

    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    public: 
        virtual Point operator()(NodeType n, double t) const {
            (void) t;
            static constexpr double grav = 9.81;

            
            return Point(0, 0, n.value().mass * -grav);  
        }

};

/**
 * @brief Damping Force.
 * 
 */
class DampingForce : public Force {
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    public:
        /**
         * @brief Damping Force -c * node.velocity.
         * 
         * @param n 
         * @param t 
         * @return Point 
         */
        virtual Point operator()(NodeType n, double t) const {
            (void) t;
            double c = 0.1;
            Point velocity =  n.value().vel;
            Point force = -c * velocity;
            return force;
        }
};

/**
 * @brief Implements force due to mass spring.
 * 
 */
class MassSpringForce : public Force {

    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    public: 
        /**
         * @brief spring force summing.
         * 
         * @param n 
         * @param t 
         * @return Point 
         */
        virtual Point operator()(NodeType n, double t) const {
            (void) t;


            Point force = Point(0, 0, 0);
            double K = 100;

            for(auto it = n.edge_begin(); it != n.edge_end(); ++it) {
                auto e = *it;
                auto dst = e.node2();
                assert(n != dst);

                // difference is a vector pointing from start to the destination
                Point difference = n.position() - dst.position();

                // length is calculated
                double current_length = e.length();

                EdgeData edge_val = e.value();
                double rest_length = edge_val.rest_length;

                Point summand = (difference / current_length) * (current_length - rest_length);
                force += -K * summand;

            }
            return force;
        }
};

/**
 * @brief Combined Force is a force and is the combination
 * of its underlings.
 * 
 */
class CombinedForce : public Force {
    public:
        /**
         * @brief returns combines sum of forces.
         * 
         * @param n Node
         * @param t double timestep
         * @return Point 
         */
        virtual Point operator()(NodeType n, double t) const {
            Point p = Point(0);

            for(Force* f_ptr : forces_) {
                auto &f = *f_ptr;
                p += f(n, t);

            }
            return p;  
        }


        /**
         * @brief Construct a new Combined Force object
         * 
         * @param forces_ 
         */
        CombinedForce(std::vector<Force*> forces_)  {
            this -> forces_ = forces_;
        }

    private:
       std::vector<Force*> forces_;
};

/**
 * @brief make_combined force from 2-3 forces.
 * 
 * can take in any type of force.
 * 
 * @param f1 
 * @param f2 
 * @param f3 
 * @return CombinedForce 
 */
CombinedForce make_combined_force(Force&& f1, Force&& f2, Force&& f3 = Force()) {
    std::vector<Force*> forces;
    forces.push_back(&f1);
    forces.push_back(&f2);
    forces.push_back(&f3);
    return CombinedForce(forces);
}