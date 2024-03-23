#include "lock_free_queue.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

TEST(Correctness, push) {
    handy::LockFreeQueue<int> queue(2);
    ASSERT_TRUE(queue.push(2));
    ASSERT_TRUE(queue.push(2));
    ASSERT_FALSE(queue.push(2));
    ASSERT_FALSE(queue.push(2));
}

TEST(Correctness, pop) {
    int value;
    handy::LockFreeQueue<int> queue(2);
    ASSERT_FALSE(queue.pop(value));
    ASSERT_FALSE(queue.pop(value));
}

TEST(Correctness, pushpop) {
    int value = 0;
    handy::LockFreeQueue<int> queue(2);
    ASSERT_TRUE(queue.push(1));
    ASSERT_TRUE(queue.pop(value));
    ASSERT_EQ(value, 1);
    ASSERT_FALSE(queue.pop(value));

    ASSERT_TRUE(queue.push(2));
    ASSERT_TRUE(queue.push(3));
    ASSERT_FALSE(queue.push(4));

    ASSERT_TRUE(queue.pop(value));
    ASSERT_EQ(value, 2);
    ASSERT_TRUE(queue.pop(value));
    ASSERT_EQ(value, 3);

    ASSERT_FALSE(queue.pop(value));
}

TEST(Performance, NoSpuriousFails) {
    const int n = 1024 * 1024;
    const int n_threads = 8;
    handy::LockFreeQueue<int> queue(n * n_threads);

    std::vector<std::thread> writers;
    // check that all threads managed to push elements without fake fails
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
    // check that all threads managed to pop elements without fake fails
    // and no elements were lost
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

TEST(Performance, NoQueueLock) {
    const int n = 1024 * 1024;
    const int n_threads = 8;
    handy::LockFreeQueue<int> queue(64);

    std::vector<std::thread> threads;
    int id = 0;
    for (int i = 0; i < n_threads; i++) {
        const int current_id = id++;
        if (current_id % 2 == 0) {
            threads.emplace_back([&] {
                for (int cnt = 0; cnt < n; ++cnt) {
                    queue.push(0);
                }
            });
        } else {
            threads.emplace_back([&] {
                for (int cnt = 0; cnt < n; ++cnt) {
                    int _;
                    queue.pop(_);
                }
            });
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    int k;
    // empty queue
    while (queue.pop(k)) {
        queue.pop(k);
    }
    // check that it still works
    ASSERT_TRUE(queue.push(0));
    ASSERT_TRUE(queue.pop(k));
    ASSERT_EQ(k, 0);
    // must be empty
    ASSERT_FALSE(queue.pop(k));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
