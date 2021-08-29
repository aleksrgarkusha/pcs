#include "catch.hpp"

#include <thread_pool.hpp>

TEST_CASE("Thread pool tests" ) {
  pcs::ThreadPool pool(std::thread::hardware_concurrency());
  CHECK(pool.num_threads() == std::thread::hardware_concurrency());
  
  pool.stop();
  CHECK_THROWS_AS(pool.add_task([]{ return true; }), std::runtime_error);

  pcs::ThreadPool default_pool;
  CHECK(std::thread::hardware_concurrency() == default_pool.num_threads());
}