#pragma once

#include <atomic>
#include <vector>

namespace handy {

/*
    Node consists of the value itself and index of the vector where it was added
    this index is required to solve ABA problem
*/
template<typename T>
struct Node {
    T value;
    // generation is changed on each assignment and then used to determine
    // whether this Node was changed by different thread or not
    std::atomic<int> generation{0};
};

template<typename T>
class LockFreeQueue {
  public:
    explicit LockFreeQueue(int size);

    LockFreeQueue(LockFreeQueue&& other) = delete;
    LockFreeQueue(const LockFreeQueue& other) = delete;

    bool push(const T& value);
    bool pop(T& data);

  private:
    std::vector<Node<T>> data_;
    std::atomic<int> head_ = 0;
    std::atomic<int> tail_ = 0;
};
}  // namespace handy
