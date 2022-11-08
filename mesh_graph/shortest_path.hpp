/**
 * @file shortest_path.cpp
 * Implimentation file for using our templated Graph to determine shortest paths.
 */


#include <vector>
#include <fstream>
#include <queue>

#include "CME212/SFML_Viewer.hpp"
#include "CME212/Util.hpp"
#include "CME212/Color.hpp"

#include "Graph.hpp"

// Define our types
using GraphType = Graph<int, double>;
using NodeType  = typename GraphType::node_type;
using NodeValueType = typename GraphType::node_value_type;
using EdgeType = typename GraphType::edge_type;
using NodeIter  = typename GraphType::node_iterator;

/**
 * @brief Functor to compare two nodes to see which is close to a given point
 * 
 */
struct NodeEuclideanComparison {

  /**
   * @brief Construct a new Node Euclidean Comparison object
   * 
   * @param pt 
   * 
   */
  NodeEuclideanComparison(const Point& pt) : p(pt) {

  }

  /**
   * @brief 
   * 
   * @param node1 
   * @param node2 
   * @return true if norm(node1.position() - p) < norm(node2.position() - p)
   * @return false if norm(node1.position() - p) > norm(node2.position() - p)
   */
  bool operator()(const NodeType& node1, const NodeType& node2) const {
    return get_euclidean_distance(node1.position(), p) <\
     get_euclidean_distance(node2.position(), p);
  }


  /** 
   * get_euclidean_distance
   * 
   * @param[in] pt1: Point&
   * @param[in] pt2: Point&
   * @return distance: double the euclidean distance between two points
   */
  double get_euclidean_distance(const Point& pt1, const Point& pt2) const {
    return norm(pt1 - pt2);
  }

  const Point& p;
};

/** Find the node with the minimum euclidean distance to a point.
 * @param g  The graph of nodes to search.
 * @param point  The point to use as the query.
 * @return An iterator to the node of @a g with the minimun Eucliean
 *           distance to @a point.
 *           graph.node_end() if graph.num_nodes() == 0.
 *
 * @post For all i, 0 <= i < graph.num_nodes(),
 *          norm(point - *result) <= norm(point - g.node(i).position())
 */
NodeIter nearest_node(const GraphType& g, const Point& point)
{

  NodeEuclideanComparison comparator(point);
  return std::min_element(g.node_begin(), g.node_end(), comparator);
}

/** Update a graph with the shortest path lengths from a root node.
 * @param[in,out] g     Input graph
 * @param[in,out] root  Root node to start the search.
 * @return The maximum path length found.
 *
 * @post root.value() == 0
 * @post Graph has modified node values indicating the minimum path length
 *           to the root.
 * @post Graph nodes that are unreachable from the root have value() == -1.
 *
 * This sets all nodes' value() to the length of the shortest path to
 * the root node. The root's value() is 0. Nodes unreachable from
 * the root have value() -1.
 */
int shortest_path_lengths(GraphType& g, NodeType& root)
{


  (void) g;


  // Basic BFS Implementation
  std::queue<NodeType> q;
  std::unordered_set<unsigned int> visited_vertices({ });
  auto val = root.value();
  val = 0;
  q.push(root);

  int dist = 0;
  while (q.size() != 0) {
    NodeType& node = q.front();

    // get node value. Use this to update it's children's root values.
    const NodeValueType& root_val =  node.value();
    q.pop();

    for(auto it = node.edge_begin(); it != node.edge_end(); ++it) {
      EdgeType adjacent = *it;

      NodeType w = adjacent.node2();
      auto got = visited_vertices.find (w.index());

      // Have not visited this vertex before
      if (got == visited_vertices.end()) {

        // Grab reference to child value 
        NodeValueType& w_val = w.value();

        // Update reference to be that of the parent's value + 1.
        w_val = root_val + 1;
        visited_vertices.emplace(w.index());

        // add to frontier of exploration.
        q.push(w);

        if (w_val > dist) {
          dist = w_val;
        }
      }

    }
  }
  return dist;
}

