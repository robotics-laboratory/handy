#include "lock_free_queue_impl.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>

using namespace handy;

TEST(Correctness, push) {
    LockFreeQueue<int> queue(2);
    ASSERT_TRUE(queue.push(2));
    ASSERT_TRUE(queue.push(2));
    ASSERT_FALSE(queue.push(2));
    ASSERT_FALSE(queue.push(2));
}

TEST(Correctness, pop) {
    int val;
    LockFreeQueue<int> queue(2);
    ASSERT_FALSE(queue.pop(val));
    ASSERT_FALSE(queue.pop(val));
}

TEST(Correctness, pushpop) {
    int val = 0;
    LockFreeQueue<int> queue(2);
    ASSERT_TRUE(queue.push(1));
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 1);
    ASSERT_FALSE(queue.pop(val));

    ASSERT_TRUE(queue.push(2));
    ASSERT_TRUE(queue.push(3));
    ASSERT_FALSE(queue.push(4));

    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 2);
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 3);

    ASSERT_FALSE(queue.pop(val));
}

TEST(Correctness, NoSpuriousFails) {
    const int n = 1024 * 1024;
    const int n_threads = 4;
    LockFreeQueue<int> queue(n * n_threads);

    std::vector<std::thread> writers;
    for (int i = 0; i < n_threads; i++) {
        writers.emplace_back([&] {
            for (int j = 0; j < n; ++j) {
                ASSERT_TRUE(queue.push(0));
            }
        });
    }

    for (auto& t : writers) {
        t.join();
    }

    std::vector<std::thread> readers;
    for (int i = 0; i < n_threads; i++) {
        readers.emplace_back([&] {
            for (int j = 0; j < n; ++j) {
                int k;
                ASSERT_TRUE(queue.pop(k));
            }
        });
    }

    for (auto& t : readers) {
        t.join();
    }
}

TEST(Correctness, NoQueueLock) {
    const int n = 1024 * 1024;
    const int n_threads = 8;
    LockFreeQueue<int> queue(64);

    std::vector<std::thread> threads;
    std::atomic<int> ids = {0};
    for (int i = 0; i < n_threads; i++) {
        threads.emplace_back([&] {
            int id = ids++;
            if (id % 2) {
                for (int j = 0; j < n; ++j) {
                    queue.push(0);
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    int k;
                    queue.pop(k);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    int k;
    while (queue.pop(k)) {
        queue.pop(k);
    }
    ASSERT_TRUE(queue.push(0));
    ASSERT_TRUE(queue.pop(k));
    ASSERT_EQ(k, 0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}