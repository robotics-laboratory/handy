#pragma once

#include <atomic>
#include <vector>

namespace handy {
template<typename T>
struct Node {
    T value;
    std::atomic<int> gen{0};
};

template<typename T>
class LockFreeQueue {
  public:
    explicit LockFreeQueue(int size) {
        max_size_ = size;
        data_ = std::vector<Node<T>>(max_size_);
        for (int i = 0; i < size; ++i) {
            data_[i].gen.store(i);
        }
    }

    LockFreeQueue() = default;
    LockFreeQueue(LockFreeQueue&& other) = delete;
    LockFreeQueue(const LockFreeQueue& other) = delete;

    int getSize(int tail, int head) {
        if (head <= tail) {
            return tail - head;
        }
        return tail + (max_size_ - head);
    }

    bool push(const T& value) {
        // while could not proccess
        while (1) {
            if (getSize(tail_.load() % max_size_, head_.load() % max_size_) == max_size_) {
                return false;
            }

            int snap = tail_.load();
            if (data_[snap % max_size_].gen < snap) {
                return false;
            }
            if (!tail_.compare_exchange_weak(snap, snap + 1)) {
                continue;
            }
            data_[snap % max_size_].value = value;
            data_[snap % max_size_].gen.fetch_add(1);
            return true;
        }
        return true;
    }

    bool pop(T& data) {
        while (1) {
            if (getSize(tail_.load(), head_.load()) == 0) {
                return false;
            }

            int snap = head_.load();
            if (data_[snap % max_size_].gen <= snap) {
                return false;
            }
            if (!head_.compare_exchange_weak(snap, snap + 1)) {
                continue;
            }

            data = data_[snap % max_size_].value;
            data_[snap % max_size_].gen.store(snap + max_size_);
            return true;
        }
        return true;
    }

  private:
    // head ------- tail
    int max_size_ = 0;
    std::vector<Node<T>> data_;
    std::atomic<int> head_ = 0;
    std::atomic<int> tail_ = 0;
};
}  // namespace handy