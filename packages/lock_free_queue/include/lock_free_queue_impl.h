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
    explicit LockFreeQueue(int size) {
        data_ = std::vector<Node<T>>(size);
        for (int i = 0; i < size; ++i) {
            data_[i].generation.store(i);
        }
    }

    LockFreeQueue(LockFreeQueue&& other) = delete;
    LockFreeQueue(const LockFreeQueue& other) = delete;

    bool push(const T& value) {
        // while could not proccess
        while (true) {
            int snap = tail_.load();
            if (snap - head_.load() == data_.size()) {
                // buffer is full and can't be updated
                // in fact, slot can be freed during verification, but we do not double-check
                return false;
            }

            if (data_[snap % data_.size()].generation < snap ||
                !tail_.compare_exchange_weak(snap, snap + 1)) {
                // desired cell in buffer was already used by another thread
                // let's try again
                continue;
            }

            data_[snap % data_.size()].value = value;
            // next possible push will be at (current_tail + 1) minimum
            // so we add +1
            data_[snap % data_.size()].generation += 1;
            return true;
        }
    }

    bool pop(T& data) {
        while (true) {
            int snap = head_.load();
            if (tail_.load() - snap == 0) {
                // buffer is empty and can't be updated
                // in fact, slot can be freed during verification, but we do not double-check
                return false;
            }

            if (data_[snap % data_.size()].generation <= snap) {
                return false;
            }
            if (!head_.compare_exchange_weak(snap, snap + 1)) {
                // desired cell in buffer was already used by another thread
                // let's try again
                continue;
            }

            data = data_[snap % data_.size()].value;
            // store a value that is for sure larger that any tail, so, + size of buffer
            data_[snap % data_.size()].generation.store(snap + data_.size());
            return true;
        }
    }

  private:
    // head ------- tail
    std::vector<Node<T>> data_;
    std::atomic<int> head_ = 0;
    std::atomic<int> tail_ = 0;
};
}  // namespace handy
