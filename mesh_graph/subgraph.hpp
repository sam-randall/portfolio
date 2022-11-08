/**
 * @file subgraph.hpp
 * Implimentation file for viewing a subgraph from our Graph
 */


#include <fstream>
#include <iterator>

#include "CME212/SFML_Viewer.hpp"
#include "CME212/Util.hpp"

#include "Graph.hpp"

/** An iterator that skips over elements of another iterator based on whether
 * those elements satisfy a predicate.
 *
 * Given an iterator range [@a first, @a last) and a predicate @a pred,
 * this iterator models a filtered range such that all i with
 * @a first <= i < @a last and @a pred(*i) appear in order of the original range.
 */
template <typename Pred, typename It>
class filter_iterator : private equality_comparable<filter_iterator<Pred,It>>
{
 public:
  // Get all of the iterator traits and make them our own
  using value_type        = typename std::iterator_traits<It>::value_type;
  using pointer           = typename std::iterator_traits<It>::pointer;
  using reference         = typename std::iterator_traits<It>::reference;
  using difference_type   = typename std::iterator_traits<It>::difference_type;
  using iterator_category = typename std::input_iterator_tag;


  /**
   * @brief Construct a new filter iterator object
   * 
   * @param p 
   * @param first 
   * @param last 
   * 
   * @post first -> index() <= beg_ -> index() <= last -> index
   * @post first -> index() < end_ -> index() <= last -> index
   * 
   * Iterates through and creates NodeIterators to first valid Node, 
   * and a NodeIterator end_ reference which is just beyond the 
   * last valid Node
   * 
   */
  filter_iterator(const Pred& p, const It& first, const It& last)
      : p_(p), end_(last), beg_(first) {

    int count = 0;
    int total = 0;
    for(auto it = first; it != last; ++it) {

      if(p(*it)) {
        if (count == 0) {
          beg_ = it;
        }
        
        count++;
      } else {

        end_ = it;
      }

      total++;
    }

    set_curr(beg_);
    offset_ = 0;

    size = count;

  }

  

  // HW1 #4: YOUR CODE HERE
  // Supply definitions AND SPECIFICATIONS for:

  /** Dereference Operator Defintion
   * @pre offset_ >= 0
   * @pre p_(*it_) == true && it_ != end_
   * @return value_type referenced by current iterator.
   * Performance: O(1) technically
   **/
  value_type operator*() const {
    return *it_;
  }

  /** Increments iterator
   * @pre offset_ >= 0
   * @pre p_(*it_) == true && it_ != end_
   * @post p_(*it_) == true || it_ != end_
   * @post offset_ = offset_ + 1
   * Performance: O(n) technically
   **/
  filter_iterator& operator++() {

    ++it_;
    ++offset_;

    while(it_ != end_ && !p_(*it_)) {
      ++it_;
    }

    return *this;
  }

  /** Equality Operator
   * Performance: O(1)
   **/
  bool operator==(const filter_iterator& other) const {
    return other.offset_ == offset_;
  }

  /** begin
   * @return a valid filter_iterator with valid offset and same size as current iterator.
   * */
  filter_iterator& begin() const {
    filter_iterator* it = new filter_iterator(p_,  beg_, end_, it_);
    it -> set_offset(0);
    it -> size = this -> get_size();
    return *it;
  }

  /** end 
   * @return a valid filter_iterator with valid offset and same size as current iterator.
   * */
  filter_iterator& end() const {
    filter_iterator* it = new filter_iterator(p_,  beg_, end_, it_);
    it -> set_offset(this -> get_size());
    it -> size = this -> get_size();
    return *it;
  }

 private:
  /** Private constructor
   * @params[in] p: predicate
   * @params[in] first
   * @params[in] last
   * @params[in] it : difference here is we can directly set the current iterator.
   * 
   * This constructor is used to create valid begin and end iterators of this class
   **/
  filter_iterator(const Pred& p, const It& first, const It& last, const It& it) : p_(p), it_(it), end_(last),  beg_(first) {}
  const Pred& p_;

  /** Iterator pointing at current element */
  It it_;

  /** Iterator pointing at end element */
  It end_;

  /** Iterator pointing at begin element */
  It beg_;

  /** Offset of the iterator */
  int offset_;

  /** Number of elements in container */
  int size;

  /**
   * @brief Get the size object
   * 
   * @return int 
   */
  int get_size() const {
    return this -> size;
  }

  /**
   * @brief Set the offset object
   * 
   * @param offset 
   */
  void set_offset(int offset) {
    offset_ = offset;
  }

  /**
   * @brief Set the curr object
   * 
   * @param it 
   */
  void set_curr(const It& it) {
    it_ = it;
  }


};

/** Helper function for constructing filter_iterators. This deduces the type of
 * the predicate function and the iterator so the user doesn't have to write it.
 * This also allows the use of lambda functions as predicates.
 *
 * Usage:
 * // Construct an iterator that filters odd values out and keeps even values.
 * std::vector<int> a = ...;
 * auto it = make_filtered(a.begin(), a.end(), [](int k) {return k % 2 == 0;});
 */
template <typename Pred, typename Iter>
filter_iterator<Pred,Iter> make_filtered(const Iter& it, const Iter& end,
                                         const Pred& p) {
  return filter_iterator<Pred,Iter>(p, it, end);
}


// Specify and write an interesting predicate on the nodes.
// Explain what your predicate is intended to do and test it.


/**
 * SlicePredicateEvenY filters z that are less than 0
 * This one is not debugging purposes but rather to 
 * perform a lobotomy on the skull
 * :)
 * */
struct SlicePredicateZLess0 {
  template <typename NODE>
  /** returns whether the z has a negative value
   * @return (node.position().z < 0)
   * @param[in] n
   * */
  bool operator()(const NODE& n) const {
    bool res =  (n.position().z < 0);
    return res;
  }
};




// For testing
struct SlicePredicateYEven {
  template <typename NODE>
  /** returns whether the y has a even value
   * @return (floor node.position().y is even)
   * @param[in] n
   * */
  bool operator()(const NODE& n) const {
    bool res =  ((int) n.position().y % 2 == 0);
    return res;
  }
};

struct SlicePredicateYPositive {
  template <typename NODE>
  
  /** returns whether the y has a negative value
   * @return (node.position().y > 0)
   * @param[in] n
   * */
  bool operator()(const NODE& n) const {
    bool res =  (n.position().y > 0);
    return res;
  }
};



 