/**
 * @file mass_spring.hpp
 * Implementation of mass-spring system using Graph
 */


#include <fstream>
#include <chrono>
#include <thread>

#include "CME212/Util.hpp"
#include "CME212/Color.hpp"

#include "GravityForce.hpp"
#include "PinConstraint.hpp"
#include "SphereConstraint.hpp"
#include "PlaneConstraint.hpp"
#include "TearSphereConstraint.hpp"
#include "SelfCollisionConstraint.hpp"
#include "combinedconstraint.hpp"

#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

// Gravity in meters/sec^2
static constexpr double grav = 9.81;

// Define the Graph type
using GraphType = Graph<NodeData, EdgeData>;
using Node = typename GraphType::node_type;
using Edge = typename GraphType::edge_type;


struct update_position {
	__host__ __device__
  using node_value_type = NodeData;
	void operator()(Node n) const {

    node_value_type& val = n.value();
    auto velocity = val.vel;

    auto update =  velocity * dt_;
		n.position() += update;
 	}

   update_position(double dt) {
     dt_ = dt;
   }

   private:
    double dt_;
};


  struct init_node {
	__host__ __device__
	void operator()(Node& node) {

    node.value().mass = 1.0 / (double) graph_ -> num_nodes();
    node.value().vel = Point(0);
    node.value().is_a_fixed_point = false;
 	}


  init_node(double dt, const GraphType* graph) {
    dt_ = dt;
    graph_ = graph;
  } 

  private:
    double dt_;
    const GraphType* graph_;
 };


struct update_velocity {
  using node_value_type = NodeData;
	__host__ __device__
  void operator()(Node n) const {
		double mass = n.value().mass;
    Point f = force_(n, t_);

    // v^{n+1} = v^{n} + F(x^{n+1},t) * dt / m
    if (!n.value().is_a_fixed_point) {
      if (n.position() == Point(0)) {
        std::cout << "Found zero and it's not a fixed point!" << std::endl;
      }
      auto new_vel = n.value().vel + f * (dt_ / mass);
      node_value_type& velocity_ref = n.value();
      velocity_ref.vel = new_vel;
    } else {
      node_value_type& velocity_ref = n.value();
      velocity_ref.vel = Point(0);
    }

 	}


  update_velocity(double dt, double t, Force& force) : force_(force) {
    dt_ = dt;
    t_ = t;
  } 

  private:
    double dt_;
    double t_;
    Force& force_;
 };



 /**
 * @brief applies the current force to the graph, given constraints.
 * @post G has nodes in correct position with velocities recorded.
 * @tparam G Graph
 * @tparam F Force
 * @tparam C Constraint
 * @param g 
 * @param t 
 * @param dt 
 * @param force 
 * @param constraint 
 * @return double timestep
 */
template <typename G, typename F, typename C>
double symp_euler_step_parallel(G& g, double t, double dt, F force, C constraint) {
  // Apply Constraints to where we need zero velocity.
  thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), update_position(dt));
  // std::cout << "success For each update position" << std::endl;
  // Apply the constraint.
  constraint(g, t);

  // Compute the t+dt velocity

 

  thrust::for_each(thrust::omp::par, g.node_begin(), g.node_end(), update_velocity(dt, t, force));
  return t + dt;

}

/**
 * @brief applies the current force to the graph, given constraints.
 * @post G has nodes in correct position with velocities recorded.
 * @tparam G Graph
 * @tparam F Force
 * @tparam C Constraint
 * @param g 
 * @param t 
 * @param dt 
 * @param force 
 * @param constraint 
 * @return double timestep
 */
template <typename G, typename F, typename C>
double symp_euler_step(G& g, double t, double dt, F force, C constraint) {
  // Apply Constraints to where we need zero velocity.

  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    auto n = *it;

    // Update the position of the node according to its velocity
    // x^{n+1} = x^{n} + v^{n} * dt
    n.position() += n.value().vel * dt;
  }

  // Apply the constraint.
  constraint(g, t);

  // Compute the t+dt velocity
  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    auto n = *it;

    

    // v^{n+1} = v^{n} + F(x^{n+1},t) * dt / m
    if (!n.value().is_a_fixed_point) {
      double mass = n.value().mass;
      Point f = force(n, t);
      n.value().vel += f * (dt / mass);
    }
  }

  return t + dt;

}
/** Change a graph's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] g      Graph
 * @param[in]     t      The current time (useful for time-dependent forces)
 * @param[in]     dt     The time step
 * @param[in]     force  Function object defining the force per node
 * @return the next time step (usually @a t + @a dt)
 *
 * @tparam G::node_value_type supports ???????? YOU CHOOSE
 * @tparam F is a function object called as @a force(n, @a t),
 *           where n is a node of the graph and @a t is the current time.
 *           @a force must return a Point representing the force vector on
 *           Node n at time @a t.
 */
template <typename G, typename F>
double symp_euler_step(G& g, double t, double dt, F force) {
  // Compute the t+dt position
  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    auto n = *it;

    // Update the position of the node according to its velocity
    // x^{n+1} = x^{n} + v^{n} * dt
    n.position() += n.value().vel * dt;
  }



  // Compute the t+dt velocity
  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
    auto n = *it;

    // v^{n+1} = v^{n} + F(x^{n+1},t) * dt / m
    n.value().vel += force(n, t) * (dt / n.value().mass);
  }



  return t + dt;
}


/** Force function object for HW2 #1. */
struct Problem1Force {
  /** Return the force applying to @a n at time @a t.
   *
   * For HW2 #1, this is a combination of mass-spring force and gravity,
   * except that points at (0, 0, 0) and (1, 0, 0) never move. We can
   * model that by returning a zero-valued force. */
  template <typename NODE>
  Point operator()(NODE n, double t) {

    (void) t;
    double K = 100;


    if (n.position() == Point(0, 0, 0) || n.position() == Point(1, 0, 0)) {
      return Point(0);
    } else {
      double mass = n.value().mass;
      Point force = Point(0, 0, -mass * grav);


      int count = 0;
      for(auto it = n.edge_begin(); it != n.edge_end(); ++it) {
        auto e = *it;
        auto dst = e.node2();
        count++;

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
  }
};



