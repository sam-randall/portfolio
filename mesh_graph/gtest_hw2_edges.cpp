#include <gtest/gtest.h>
#include <fstream>

#include "CME212/Util.hpp"
#include "CME212/Point.hpp"

#include "Graph.hpp"
#include "mass_spring.hpp"

class GraphPointFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<NodeData,EdgeData>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  Point p1, p2;

  virtual void SetUp() {
    p1 = Point(CME212::random(), CME212::random(), CME212::random());
    p2 = Point(CME212::random(), CME212::random(), CME212::random());
  }
};


// test Edge Size
TEST_F(GraphPointFixture, EdgeSize){
  EXPECT_LE( sizeof(EdgeType), 32 )  << "edge size > 32 bytes";
}

// Test add edge
TEST_F(GraphPointFixture, AddEdge){
  
  //Inserting Node
  graph.add_node(p1);
  graph.add_node(p2);
  //Graph has 2 Nodes
  EXPECT_EQ(graph.num_nodes(),2) << "Graph does not have 2 Nodes";
  
  //Inserting Edge
  EdgeType e = graph.add_edge(graph.node(0), graph.node(1));
  //Graph has 1 Edge
  EXPECT_EQ(graph.num_edges(),1) << "Graph does not have 1 Edge";
  //Getting Edge
  EXPECT_EQ(e, graph.edge(0)) << "Edge equality or get error ";
}


// Test adding node with default value
TEST_F(GraphPointFixture, NodePositionChange){

  NodeType n0 = graph.add_node(p1);
  NodeType ref = graph.node(0);
  n0.position() = Point(0, 4, 0);
  EXPECT_EQ(ref.position(), Point(0, 4, 0)) << "Error";
  
  NodeType n1 = graph.add_node(p2);

  EdgeType e = graph.add_edge(n0, n1);

  e.node1().position() = Point(3);

  EXPECT_EQ(e.node1().position(), Point(3)) << "Error with edge node 1 update";
  EXPECT_TRUE(ref.position() == Point(3)) << "Error: first node not updated.";

}

// Test adding node with default value
TEST_F(GraphPointFixture, NodePositionChange2){

  NodeType n0 = graph.add_node(p1);
  NodeType ref = graph.node(0);
  n0.position() = Point(0, 4, 0);
  EXPECT_EQ(ref.position(), Point(0, 4, 0)) << "Error";
  
  NodeType n1 = graph.add_node(p2);
  NodeType ref2 = graph.node(1);

  ref2.position() = Point(99);
  EXPECT_EQ(ref2.position(), Point(99)) << "Error";
  EXPECT_EQ(n1.position(), Point(99)) << "Error";

}

class SingleEdgeFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int,int>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  std::vector<Point> points;
  Point p1, p2;
  // EdgeType e;

  virtual void SetUp() {
    p1 = Point(CME212::random(), CME212::random(), CME212::random());
    p2 = Point(CME212::random(), CME212::random(), CME212::random());
    graph.add_node(p1);
    graph.add_node(p2);
    graph.add_edge(graph.node(0), graph.node(1));
    // EdgeType e = graph.edge(0);
  }
};


class DoubleEdgeFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int,int>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  std::vector<Point> points;
  Point p1, p2, p3;
  // EdgeType e;

  virtual void SetUp() {
    p1 = Point(CME212::random(), CME212::random(), CME212::random());
    p2 = Point(CME212::random(), CME212::random(), CME212::random());
    p3 = Point(CME212::random(), CME212::random(), CME212::random());
    graph.add_node(p1);
    graph.add_node(p2);
    graph.add_node(p3);
    graph.add_edge(graph.node(0), graph.node(1));
    graph.add_edge(graph.node(1), graph.node(2));
    // EdgeType e = graph.edge(0);
  }
};

// Test has_edge
TEST_F(DoubleEdgeFixture, HasEdge){
  EdgeType e = graph.edge(0);
  //Graph has_edge e
  EXPECT_TRUE(graph.has_edge(e.node1(), e.node2())) << "Graph does not have edge e";
  //Graph has_edge e transpose
  EXPECT_TRUE(graph.has_edge(e.node2(), e.node1())) << "Graph does not have edge e transpose";
  EdgeType e1 = graph.edge(1);
  //Graph has_edge e
  EXPECT_TRUE(graph.has_edge(e1.node1(), e1.node2())) << "Graph does not have edge e";
  //Graph has_edge e transpose
  EXPECT_TRUE(graph.has_edge(e1.node2(), e1.node1())) << "Graph does not have edge e transpose";

}

TEST_F(DoubleEdgeFixture, RemoveEdgeLoop){

  auto node = graph.node(0);


  unsigned int count = 0;
  auto degree = node.degree();

  for(auto it = node.edge_begin(); count < degree;) {

    auto e = *it;
    bool didRemove = graph.remove_edge(e);

    EXPECT_TRUE(didRemove) << "Failed to remove edge in loop";

    ++it;
    count++;

  }


  auto node1 = graph.node(1);
  auto it = node1.edge_begin();


  auto node2 = graph.node(2);

  auto e = *it; 


  EXPECT_EQ(e.node1(), node1 ) << "Not node 1";
  EXPECT_EQ(e.node2(),node2 ) << "Not node 2";



  EXPECT_EQ(graph.num_edges(), 1) << "FAiled to remove edge.";
}



// Test has_edge
TEST_F(SingleEdgeFixture, HasEdge){
  EdgeType e = graph.edge(0);
  //Graph has_edge e
  EXPECT_TRUE(graph.has_edge(e.node1(), e.node2())) << "Graph does not have edge e";
  //Graph has_edge e transpose
  EXPECT_TRUE(graph.has_edge(e.node2(), e.node1())) << "Graph does not have edge e transpose";
}


// Test nodes of edge
TEST_F(SingleEdgeFixture, EdgeCheck){
  EdgeType e = graph.edge(0);
  //Edge nodes check out
  EXPECT_TRUE(
          (e.node1() == graph.node(0) && e.node2() == graph.node(1)) ||
          (e.node1() == graph.node(1) && e.node2() == graph.node(0)) 
  ) << "Edge nodes do not check out";
}

// Test edge removal
TEST_F(SingleEdgeFixture, RemoveEdge){
  EdgeType e = graph.edge(0);

  //remove_edge return value is true
  EXPECT_TRUE(graph.remove_edge(e.node1(), e.node2())) << "remove_edge return value is false";
  //Graph has 0 edges


  EXPECT_EQ(graph.num_edges(),0) << "Graph does not have 0 edges";
  //Graph !has_edge

  EXPECT_FALSE(graph.has_edge(graph.node(0), graph.node(1))) << "Graph has_edge that was removed";
  //remove_edge return value is false

  EXPECT_FALSE(graph.remove_edge(graph.node(0), graph.node(1))) << "remove_edge return value is true";
}


// Test add_edge twice after removal
TEST_F(SingleEdgeFixture, AddEdgeAfter){
  EdgeType e = graph.edge(0);

  //remove edge
  graph.remove_edge(e.node1(), e.node2());
  // Inserting Edge

  graph.add_edge(graph.node(0), graph.node(1));
  //Inserting Edge Again
  graph.add_edge(graph.node(0), graph.node(1));
  //Graph has 1 Edge
  EXPECT_EQ(graph.num_edges(),1) << "Graph adds duplicate edges";
}


// Test add_edge twice after removal
TEST_F(DoubleEdgeFixture, RemoveEdgeCheckEdgeIndex){
  EdgeType e = graph.edge(0);


  //remove edge
  graph.remove_edge(e.node1(), e.node2());
  // Inserting Edge

  EdgeType e2 = graph.edge(0);

  EXPECT_EQ(e2.index(), 0) << "Unexpected edge index";
}

// Test Edges after node removal
TEST_F(SingleEdgeFixture, EdgeAfterNodeRemove){


  //Removing Node ...
  EXPECT_TRUE(graph.remove_node(graph.node(1))) << "remove_node is false";
  //Graph has 1 node
  EXPECT_EQ(graph.num_nodes(),1) <<  "Graph does not have 1 node";
  //Edge removed b/c of Node


  EXPECT_EQ(graph.num_edges(),0) << "Edge not removed after node removal";
}

// Test Clearing
TEST_F(SingleEdgeFixture,Clearing){

  //Clear
  graph.clear();
  //Graph has no edges
  EXPECT_EQ(graph.num_edges(),0) << "Graph has edges after clearing";
}


class HundredEdgeFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int,int>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  // EdgeType e;

  virtual void SetUp() {
    //Add 100 Nodes
    for (int k = 0; k < 100; ++k) {
      graph.add_node(Point(CME212::random(), CME212::random(), CME212::random()));
    }
    // Adding 100 Edges
    for (unsigned k = 0; k < 100; ++k) {
      unsigned n1, n2;
      do {
        n1 = (unsigned) CME212::random(0, graph.num_nodes());
        n2 = (unsigned) CME212::random(0, graph.num_nodes());
      } while (n1 == n2 || graph.has_edge(graph.node(n1), graph.node(n2)));

        graph.add_edge(graph.node(n1), graph.node(n2));
      }
    }
};

//Test multiple edge removal
TEST_F(HundredEdgeFixture, RepeatedRemoveEdge){

  // Remove 50 Edges
  for (unsigned k = 0; k < 50; ++k) {
    unsigned n1, n2;
    do {
      n1 = (unsigned) CME212::random(0, graph.num_nodes());
      n2 = (unsigned) CME212::random(0, graph.num_nodes());
    } while (!graph.has_edge(graph.node(n1), graph.node(n2)));

    graph.remove_edge(graph.node(n1), graph.node(n2));

    if (k == 23){
      // Graph !has_edge after remove
      EXPECT_FALSE(graph.has_edge(graph.node(n1), graph.node(n2))) << "Graph has_edge after remove";
    }
  }
  //Removed 50 Edges
  EXPECT_EQ(graph.num_edges(), 50) << "Did not remove 50 Edges";

  // Count edges the long way
  unsigned count_edges = 0;
  for (unsigned k = 0; k < graph.num_nodes(); ++k) {
    for (unsigned j = k+1; j < graph.num_nodes(); ++j) {
      if (graph.has_edge(graph.node(k), graph.node(j)))
        ++count_edges;
    }
  }
  
  //Edge count agrees
  EXPECT_EQ(graph.num_edges(), count_edges) << "Edge count does not agree";
}


// Test repeated remove node
TEST_F(HundredEdgeFixture, RepeatedRemoveNodeNonRandomIndex){
  // Remove 50 Nodes...

  assert(graph.node(5).index() == 5);

  graph.remove_node(graph.node(5));

  NodeType n = graph.node(5);
  EXPECT_EQ(n.index(), 5) << "Index not expected value after remove.";
}


// Test repeated remove node
TEST_F(HundredEdgeFixture, RepeatedRemoveNodeNonRandom){
  // Remove 1 Nodes...
  assert(graph.node(5).index() == 5);
  graph.remove_node(graph.node(5));
}


// Test repeated remove node
TEST_F(HundredEdgeFixture, RepeatedRemoveNode){
  // Remove 50 Nodes...
  for (unsigned k = 0; k < 50; ++k) {
      unsigned n = (unsigned) CME212::random(0, graph.num_nodes());
      graph.remove_node(graph.node(n));
  }
  // removed 50 Nodes
  EXPECT_EQ(graph.num_nodes(), 50) << "Did not remove 50 Nodes";

  // Count edges the long way
  unsigned count_edges = 0;
  for (unsigned k = 0; k < graph.num_nodes(); ++k) {
    for (unsigned j = k+1; j < graph.num_nodes(); ++j) {
      if (graph.has_edge(graph.node(k), graph.node(j)))
        ++count_edges;
    }
  }


  //Edge count agrees
  EXPECT_EQ(count_edges, graph.num_edges()) << "Edge count does not agree";

}


class TwoGraphFixture  : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int,int>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Nodes
  GraphType g, g2;
};

// Test comparisons
TEST_F(TwoGraphFixture, CompareEdges){

  //Adding 10 Nodes to Graph1 and Graph2...
  for (unsigned k = 0; k < 10; ++k) {
    Point p(CME212::random(), CME212::random(), CME212::random());
    g.add_node(p);
    g2.add_node(p);
  }

  EdgeType e1 = g.add_edge(g.node(3), g.node(4));
  EdgeType e2 = g2.add_edge(g2.node(3), g2.node(4));
  
  //E1-E1 Edge comparison ==
  EXPECT_EQ(e1,e1) << "E1-E1 Edge comparison == error";
  //G1-G2 Edge comparison !=
  EXPECT_NE(e1, e2) << "G1-G2 Edge comparison != error";
  //G1-G2 Edge comparison < >
  EXPECT_TRUE(e1 < e2 || e2 < e1) << "G1-G2 Edge comparison < > error";
}


