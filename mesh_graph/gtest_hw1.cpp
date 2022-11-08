#include <gtest/gtest.h>
#include <fstream>

#include "CME212/Util.hpp"
#include "Graph.hpp"
#include "shortest_path.hpp"
#include "subgraph.hpp"



class GraphPointFixture : public ::testing::Test {
 protected:
   //Define types
  using GraphType = Graph<int, double>;
  using NodeType  = typename GraphType::node_type;
  using NodeIter = typename GraphType::node_iterator;
  using EdgeType  = typename GraphType::edge_type;

  //Set up Graph and Points
  GraphType graph;
  std::vector<Point> points;
  virtual void SetUp() {
    for(int i = 0; i < 10; i++)
      points.push_back(Point(i));
  }
  
};

// Test adding node with default value
TEST_F(GraphPointFixture, DefaultNodeVal){
  graph.add_node(points[0]);
  EXPECT_EQ( graph.node(0).value(), 0 ) << "add_node does not intalize node vale with a default 0 value";
}


// Test adding node with default value
TEST_F(GraphPointFixture, AddManyNodes){

  for (int i = 0; i < 100; ++i) {
    graph.add_node(points[i % 3]);
  }
}


// Test adding node with default value
TEST_F(GraphPointFixture, NodePositionChange){

  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);

  EdgeType e = graph.add_edge(n0, n1);
  e.node1().position() = Point(4);
  EXPECT_EQ(e.node1().position(), Point(4)) << "node position";
  EXPECT_EQ(n0.position(), Point(4)) << "node position";


}


// Test degree function
TEST_F(GraphPointFixture, Degree){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);
  graph.add_edge(n0, n1);

  EXPECT_EQ(n2.degree(),0)  << "n2 degree is 0";
  EXPECT_EQ(n1.degree(), 1) << "n1 degree is 1";
}

// Test node iterator
TEST_F(GraphPointFixture, NodeBeginIter){
  NodeType n0 = graph.add_node(points[0]);
  graph.add_node(points[1]);

  auto it = graph.node_begin();
  
  EXPECT_EQ(*it, n0) << " error in node iteration " ;
}


// Test node iterator
TEST_F(GraphPointFixture, IterCheckn1){
  graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_node(points[2]);

  auto it = graph.node_begin();

  ++it;
  
  EXPECT_EQ(*it, n1) << " error in node iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, MakeFilteredExecutesSuccessfully){

  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  graph.add_node(points[3]);
  
  SlicePredicateYPositive pred;

  make_filtered(graph.node_begin(), graph.node_end(), pred);

  EXPECT_EQ(0, 0) << "Make Filtered Crash";
}

// Test node iterator
TEST_F(GraphPointFixture, FilterIteratorBeginTest){

  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  graph.add_node(points[3]);
  
  SlicePredicateYPositive pred;

  auto iter = make_filtered(graph.node_begin(), graph.node_end(), pred);

  auto it = iter.begin();


  EXPECT_TRUE(pred(*it)) << "Beginning Node does not satisfy condition." ;

}

// Test node iterator
TEST_F(GraphPointFixture, FilterIteratorTestEvenY){

  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  graph.add_node(points[3]);
  
  SlicePredicateYEven pred;
  int count = 0;
  auto iter = make_filtered(graph.node_begin(), graph.node_end(), pred);
  
  auto begin = iter.begin();
  auto end = iter.end();

  for(auto it = begin; it != end; ++it) {
    if(pred(*it) ) {
      count++;
    }
  }
  
  EXPECT_EQ(count, 2) << " error in node iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, FilterIteratorTest){

  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  graph.add_node(points[3]);
  
  EXPECT_EQ(1, 1) << "made pred";
  SlicePredicateYPositive pred;

  auto iter = make_filtered(graph.node_begin(), graph.node_end(), pred);

  int count = 0;

  auto begin = iter.begin();
  auto end = iter.end();


  int debug = 0;
  for(auto it = begin; it != end; ++it) {
    NodeType n = *it;
    if(pred(n) ) {
      count++;
    } 

    debug++;

  }
  
  EXPECT_EQ(count, 3) << " error in Filter iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorBegin){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n2);

  auto edge = n0.edge_begin();
  EdgeType e = *edge;
  NodeType n = e.node1();

  EXPECT_EQ(n, n0) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorBeginNEQ1){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n2);
  auto edge = n0.edge_begin();
  EdgeType e = *edge;
  NodeType n = e.node1();

  
  EXPECT_NE(n, n1) << " error in node incident edges iteration " ;
}


// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorBeginNEQ2){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n2);

  auto edge = n0.edge_begin();
  EdgeType e = *edge;
  NodeType n = e.node1();

  
  EXPECT_NE(n, n2) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorTestOrder){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_node(points[2]);

  graph.add_edge(n1, n0);

  auto edge = n0.edge_begin();
  EdgeType e = *edge;
  NodeType n = e.node1();
  EXPECT_EQ(n, n0) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorTestOrder2){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n1, n0);
  graph.add_edge(n1, n2);
  graph.add_edge(n2, n0);

  auto edge = n0.edge_begin();
  EdgeType e = *edge;
  NodeType n = e.node1();
  EXPECT_EQ(n, n0) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorTestOrderOneIncrement){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n1, n0);
  graph.add_edge(n1, n2);
  graph.add_edge(n2, n0);

  auto edge = n0.edge_begin();
  ++edge;
  EdgeType e = *edge;
  NodeType n = e.node1();
  EXPECT_EQ(n, n0) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorDeref){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  EdgeType e1 = graph.add_edge(n0, n1);
  EdgeType e2 = graph.add_edge(n0, n2);
  

  auto edge_it = n0.edge_begin();
  

  EdgeType e = *edge_it;
  

  EXPECT_TRUE((e == e1 || e == e2)) << "edge it did not crash" ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorIncrement){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n2);


  auto edge_it = n0.edge_begin();
  
  ++edge_it;


  EXPECT_TRUE(true) << "edge it did not crash" ;
}

// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorIncrementAndDeref){

  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  EdgeType e1 = graph.add_edge(n0, n1);
  EdgeType e2 = graph.add_edge(n0, n2);

  auto edge_it = n0.edge_begin();
  
  ++edge_it;

  EdgeType e = *edge_it;

  EXPECT_TRUE(((e == e1 || e == e2))) << "edge it did not crash" ;
}

TEST_F(GraphPointFixture, IncidentIteratorTestincrementNotequalBegin){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_edge(n0, n1);
  
  auto edge = n0.edge_begin();

  ++edge;
  EXPECT_NE(edge, n0.edge_begin()) << " error in node incident edges iteration " ;
}

TEST_F(GraphPointFixture, IncidentIteratorTestEnd){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  graph.add_edge(n0, n1);
  auto begin = n0.edge_begin();
  auto edge = begin;

  ++edge;
  EXPECT_EQ(edge, n0.edge_end()) << " error in node incident edges iteration " ;
}



// Test node iterator
TEST_F(GraphPointFixture, IncidentIteratorTest){
  NodeType n0 = graph.add_node(points[0]);
  NodeType n1 = graph.add_node(points[1]);
  NodeType n2 = graph.add_node(points[2]);

  graph.add_edge(n0, n1);
  graph.add_edge(n0, n2);
  
  int iter = 0;

  auto edge = n0.edge_begin();

  for(; edge != n0.edge_end(); ++edge) {
    iter++;
  }
  
  EXPECT_EQ(iter, 2) << " error in node incident edges iteration " ;
}

// Test node iterator
TEST_F(GraphPointFixture, NodeIter){
  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  
  int iter = 0;
  auto ni = graph.node_begin();
  
  for(; ni != graph.node_end(); ++ni){
    ++iter;
  }
  EXPECT_EQ(iter, 3) << " error in node iteration " ;
}

// Test node iterator with no points
TEST_F(GraphPointFixture, NodeIterEmpty){
  int iter = 0;
  auto ni = graph.node_begin();
  for(; ni != graph.node_end(); ++ni){
    ++iter;
  }
  EXPECT_EQ(iter, 0) << " error in node iteration with no points" ;
}

//Test nearest node
TEST_F(GraphPointFixture, ShortestPath){
  graph.add_node(points[0]);
  graph.add_node(points[1]);
  graph.add_node(points[2]);
  
  NodeIter nearest = nearest_node(graph, Point(0));
  EXPECT_EQ( *nearest, graph.node(0)) << " error finding nearest node " ;
}
