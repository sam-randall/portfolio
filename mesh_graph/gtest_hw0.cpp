#include <gtest/gtest.h>
#include <fstream>
#include "CME212/Util.hpp"
#include "Graph.hpp"

class GraphPointFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int, double>;
  using NodeType  = typename GraphType::node_type;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  std::vector<Point> points;
  virtual void SetUp() {
    for(int i = 0; i < 10; i++)
      points.push_back(Point(i));
  }
  
};

// Test has_node function
TEST_F(GraphPointFixture, HasNode){
  GraphType::node_type n0 = graph.add_node(points[0]);
  EXPECT_TRUE( graph.has_node(n0) ) << "has_node did not find n0";
}




// Test num nodes/size functions
TEST_F(GraphPointFixture, Size){
  EXPECT_EQ(graph.num_nodes(),graph.size()) << "num_nodes and size are different"  ;
  EXPECT_EQ(graph.size(), 0) << "starting size is not 0" ;

  graph.add_node(points[0]);
  graph.add_node(points[1]);

  EXPECT_EQ(graph.num_nodes(),graph.size()) << "num_nodes and size are different";
  EXPECT_EQ(graph.size(), 2) << "size is incorrect";
}

// Test edge function
TEST_F(GraphPointFixture, Edge){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_node(points[2]);    
  EdgeType e0 = graph.add_edge(n0, n1);
  
  EXPECT_EQ(e0, graph.edge(0)) << "error in edge retreval"  ;
}

// Verify only one of e0 < e1 or e1 < e0 is true
TEST_F(GraphPointFixture, Tricotomy){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);
  
  EdgeType e0 = graph.add_edge(n0, n1);
  EdgeType e1 = graph.add_edge(n1, n2);

  EXPECT_TRUE( (e0 < e1) ^ (e1 < e0) ) << "error in edge comparison";
}

TEST_F(GraphPointFixture, NodePositionX){
  NodeType n0 = graph.add_node(points[0]);
  EXPECT_NEAR(n0.position().x,  points[0].x, 1e-8) << "error in position.X return";
}

TEST_F(GraphPointFixture, NodePositionFromIndexY){
  graph.add_node(points[2]);
  NodeType n0 = graph.node(0);
  EXPECT_NEAR(n0.position().y,  points[2].y, 1e-8) << "error in position.Y  return";
}

TEST_F(GraphPointFixture, NodePositionY){
  NodeType n0 = graph.add_node(points[1]);
  EXPECT_NEAR(n0.position().y,  points[1].y, 1e-8) << "error in position.Y  return";
}

TEST_F(GraphPointFixture, NodePositionZ){
  NodeType n0 = graph.add_node(points[2]);
  EXPECT_NEAR(n0.position().z,  points[2].z, 1e-8) << "error in position.Z  return";
}

TEST_F(GraphPointFixture, NodeTricotomy){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  EXPECT_TRUE( (n0 < n1) ^ (n1 < n0) ) << "error in node comparison";
}

TEST_F(GraphPointFixture, NodeIndex){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  EXPECT_TRUE( (n0.index() == 0) &&  (n1.index() == 1)) << "error in node index";
}
 
TEST_F(GraphPointFixture, NodeSize){
  NodeType n0 = graph.add_node(points[0]);
  EXPECT_TRUE( (sizeof(n0) <= 16)) << "size: " << sizeof(n0) << ", Node Size Too Big";
}

TEST_F(GraphPointFixture, EdgeSize){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  EdgeType e0 = graph.add_edge(n0, n1);
  EXPECT_TRUE( (sizeof(e0) <= 32)) << sizeof(e0) << "Edge Size Too Big";
}

TEST_F(GraphPointFixture, EdgeVertexOrderDifferentOrder){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_edge(n0, n1);
  EdgeType e0_again = graph.add_edge(n1, n0);
  EXPECT_TRUE( (e0_again.node1() == n1) && (e0_again.node2() == n0))  << "add existing edge returns vertices in wrong order.";
}

TEST_F(GraphPointFixture, EdgeVertexOrderSameOrder){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_edge(n0, n1);
  EdgeType e0_again = graph.add_edge(n0, n1);
  EXPECT_TRUE( (e0_again.node1() == n0) && (e0_again.node2() == n1))  << "add existing edge returns vertices in wrong order.";

}


// Add existing edge doesn't create another edge
TEST_F(GraphPointFixture, AddEdge){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n1);
  graph.add_edge(n1, n0);

  EXPECT_EQ(graph.num_edges(), 1) << " add edge creates duplicate edges " ;

}

