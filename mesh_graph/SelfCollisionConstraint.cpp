

#include "Constraint.hpp"
#include "SpaceSearcher.hpp"
#include <algorithm>
#ifndef SPHERECONSTRAINT_HPP
#define SPHERECONSTRAINT_HPP


struct node_to_point : public thrust::unary_function<Node,Point> {
	__host__ __device__
	Point operator()(Node& n) const {
		return n.position();
 	}
 };
template<typename V, typename E>
Box3D graph_bounding_box(const Graph<V, E>& g) {
	auto it = thrust::make_transform_iterator(g.node_begin(), node_to_point());
	auto end = thrust::make_transform_iterator(g.node_end(), node_to_point());
	return Box3D(it, end);
}



struct check_for_collisions {
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    void operator()(NodeType node) {

        const Point& center = node.position();

        double radius2 = std::numeric_limits<double>::max();

        for(auto it = node.edge_begin(); it != node.edge_end(); ++it) {

            NodeType n2 = it -> node2();
            radius2 = std::min(radius2, normSq(n2.position() - center));
        }

        radius2 *= 0.9;

        Box3D query_box(center - radius2, center + radius2);

        for (auto it = searcher_.begin(query_box); it != searcher.end(query_box); ++it ){
            NodeType n2 = *it;
            Point r = center - n2.position()
            double l2 = normSq(r);
            if (node != n2 && l2 < radius2) {
                node.value().vel -= (dot(r, node.value().vel / l2)) * r;
            }
        }
    }


    check_for_collisions(SpaceSearcher<NodeType>& searcher) : searcher_(searcher){
        std::cout << "checking" << std::endl;
    }


    private:
        const SpaceSearcher<NodeType>& searcher_

};


/**
 * @brief 
 * 
 */
class SelfCollisionConstraint : public Constraint {

    public:
        using GraphType = Graph<NodeData, EdgeData>;
        using NodeType  = typename GraphType::node_type;

        /**
         * @brief SelfCollision Constraint
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
              // Bounding box of all the nodes
            Box3D bigbb = graph_bounding_box(g);

            // Construct the Searcher
            auto n2p = [](const NodeType& n) { return n.position(); };
            SpaceSearcher<NodeType> searcher(bigbb, g.node_begin(), g.node_end(), n2p);
            std::cout << "self collision constraint" << std::endl;
            thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), check_for_collisions(searcher));
            std::cout << "done self collision const" << std::endl;
        }
};

#endif