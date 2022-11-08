#include "Constraint.hpp"
#include "SpaceSearcher.hpp"
#include <algorithm>
#include <math.h>
#ifndef SELFCOLLISIONCONSTRAINT_HPP
#define SELFCOLLISIONCONSTRAINT_HPP

/**
 * @brief node to point functor, calculates position.
 * 
 */
struct node_to_point {
	using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;

    /**
     * @brief operator() that gets position.
     * 
     * @param n 
     * @return Point 
     */
	Point operator()(const NodeType& n) const {
		return n.position();
 	}
 };

/**
 * @brief gets a bounding box that contains all nodes in the graph.
 * 
 * @tparam V 
 * @tparam E 
 * @param g 
 * @return Box3D 
 */
template<typename V, typename E>
Box3D graph_bounding_box(const Graph<V, E>& g) {
	auto it = thrust::make_transform_iterator(g.node_begin(), node_to_point());
	auto end = thrust::make_transform_iterator(g.node_end(), node_to_point());
	auto uninclusive = Box3D(it, end);
    return uninclusive | (uninclusive.min() - 1e-2) | (uninclusive.max() + 1e-2);
}


/**
 * @brief check_for_collisions 
 * 
 */
struct check_for_collisions {
    public:
    using GraphType = Graph<NodeData, EdgeData>;
    using NodeType  = typename GraphType::node_type;
    /**
     * @brief does an optimized morton coded way of ensuring no self collisions.
     * 
     * @param node 
     */
    void operator()(NodeType node) {

        const Point& center = node.position();

        double radius2 = std::numeric_limits<double>::max();

        // Iterate through adjacent vertices and find one that is closest
        for(auto it = node.edge_begin(); it != node.edge_end(); ++it) {
            auto e = *it;
            NodeType n2 = e.node2();
            radius2 = std::min(radius2, normSq(n2.position() - center));
        }

        radius2 *= 0.9;

        Box3D bounding_box = searcher_.bounding_box();

        Point min_bb = bounding_box.min();
        Point max_bb = bounding_box.max();
        Point lower_unbounded = center - std::sqrt(radius2);
        Point upper_unbounded = center + std::sqrt(radius2);

        Point lower = Point(std::max(lower_unbounded.x, min_bb.x),
                             std::max(lower_unbounded.y, min_bb.y),
                              std::max(lower_unbounded.z, min_bb.z));
        Point upper = Point(std::min(upper_unbounded.x, max_bb.x),
                             std::min(upper_unbounded.y, max_bb.y),
                              std::min(upper_unbounded.z, max_bb.z));

        Box3D query_box(lower, upper);

        for (auto it = searcher_.begin(query_box); it != searcher_.end(query_box); ++it ){
            NodeType n2 = *it;
            Point r = center - n2.position();
            double l2 = normSq(r);
            if (node != n2 && l2 < radius2) {
                node.value().vel -= (dot(r, node.value().vel / l2)) * r;
            }
        }

    }

    /**
     * @brief Construct a new check for collisions object
     * 
     * @param searcher 
     */
    check_for_collisions(const SpaceSearcher<NodeType>& searcher) : searcher_(searcher) {}


    private:
        /** searcher loaded in */
        const SpaceSearcher<NodeType>& searcher_;

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
         * @post no nodes are arranged in such a way that there are 'self collisions'
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

            thrust::for_each(thrust::seq, g.node_begin(), g.node_end(), check_for_collisions(searcher));
        }
};

#endif