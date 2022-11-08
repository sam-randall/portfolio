#ifndef DATA_HPP
#define DATA_HPP

/** Custom structure of data to store with Nodes */
struct NodeData {
  Point vel;       //< Node velocity
  double mass;     //< Node mass
  bool is_a_fixed_point;
  NodeData() : vel(0), mass(1), is_a_fixed_point(false) {}

};

/**
 * @brief Edge Data
 * @param rest_length 
 */
struct EdgeData {
  double rest_length; // Initial length

  EdgeData() : rest_length(0){}


};

#endif