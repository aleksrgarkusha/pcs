#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pcs {

class ThreadPool {
 public:
  explicit ThreadPool(
      const int num_threads = std::thread::hardware_concurrency());
  ~ThreadPool();

  inline size_t num_threads() const;

  template <class F, class... Args>
  auto add_task(F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

  void stop();
  void wait();

 private:
  void worker_func();
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable task_condition_;
  std::condition_variable finished_condition_;

  bool stopped_;
  std::atomic<int> num_active_workers_;
};

inline size_t ThreadPool::num_threads() const { return workers_.size(); }

template <class F, class... Args>
auto ThreadPool::add_task(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_t = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared<std::packaged_task<return_t()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_t> result = task->get_future();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      throw std::runtime_error("Cannot add task to stopped thread pool.");
    }
    tasks_.emplace([task]() { (*task)(); });
  }
  task_condition_.notify_one();

  return result;
}

inline ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
  for (int index = 0; index < num_threads; ++index) {
    std::function<void(void)> worker =
        std::bind(&ThreadPool::worker_func, this);
    workers_.emplace_back(worker);
  }
}

inline ThreadPool::~ThreadPool() { stop(); }

inline void ThreadPool::stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
  }

  {
    std::queue<std::function<void()>> empty_tasks;
    std::swap(tasks_, empty_tasks);
  }
  task_condition_.notify_all();

  for (auto &worker : workers_) {
    worker.join();
  }
}

inline void ThreadPool::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_condition_.wait(
      lock, [this]() { return tasks_.empty() && num_active_workers_ == 0; });
}

inline void ThreadPool::worker_func() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_condition_.wait(lock,
                           [this] { return stopped_ || !tasks_.empty(); });
      if (!tasks_.empty()) {
        task = std::move(tasks_.front());
        tasks_.pop();
        num_active_workers_.fetch_add(1);
      } else
        return;
    }

    task();
    num_active_workers_.fetch_sub(1);
    finished_condition_.notify_one();
  }
}

}  // namespace pcs