#pragma once

#include <algorithm>
#include <atomic>
#include <limits>
#include <stack>
#include <thread>
#include <queue>

#include <Eigen/Dense>

namespace pcs {

template<uint16_t K, typename index_t = unsigned>
class KDTree {
public:
  const index_t nai_v = std::numeric_limits<index_t>::max();
public:
  KDTree(const std::vector<Eigen::Matrix<double, K, 1>> &vertices,
         int max_threads = 2 * std::thread::hardware_concurrency());

  std::pair<index_t, double>
  find_nn(Eigen::Matrix<double, K, 1> point,
          double max_dist = std::numeric_limits<double>::infinity()) const;

  std::vector<std::pair<index_t, double>>
  find_nns(Eigen::Matrix<double, K, 1> point, std::size_t n,
           double max_dist = std::numeric_limits<double>::infinity()) const;

private:
  const std::vector<Eigen::Matrix<double, K, 1>> &vertices;

  struct Node {
    typedef index_t ID;
    decltype(K) d;
    index_t first;
    index_t last;
    index_t vertex_id;
    Node::ID left;
    Node::ID right;
  };

  std::atomic<index_t> num_nodes;
  std::vector<Node> nodes;

  typename Node::ID CreateNode(decltype(K) d, index_t first, index_t last) {
    typename Node::ID node_id = num_nodes++;
    Node &node = nodes[node_id];
    node.first = first;
    node.last = last;
    node.left = nai_v;
    node.right = nai_v;
    node.vertex_id = nai_v;
    node.d = d;
    return node_id;
  }

  std::pair<typename Node::ID, typename Node::ID>
  ssplit(typename Node::ID node_id, std::vector<index_t> *indices);

  void split(typename Node::ID node_id, std::vector<index_t> *indices,
             std::atomic<int> *num_threads);
};

template<uint16_t K, typename IdxType>
KDTree<K, IdxType>::KDTree(const std::vector<Eigen::Matrix<double, K, 1>>&vertices,
                           int max_threads)
    : vertices(vertices), num_nodes(0) {

  std::size_t num_vertices = vertices.size();
  nodes.resize(num_vertices);

  std::vector<IdxType> indices(num_vertices);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  std::atomic<int> num_threads(max_threads);
  split(CreateNode(0, 0, num_vertices), &indices, &num_threads);
}

template<uint16_t K, typename IdxType>
void KDTree<K, IdxType>::split(typename Node::ID node_id,
                               std::vector<IdxType> *indices,
                               std::atomic<int> *num_threads) {
  typename Node::ID left, right;
  if ((*num_threads -= 1) >= 1) {
    std::tie(left, right) = ssplit(node_id, indices);
    if (left != nai_v && right != nai_v) {
      std::thread other(&KDTree::split, this, left, indices, num_threads);
      split(right, indices, num_threads);
      other.join();
    } else {
      if (left != nai_v)
        split(left, indices, num_threads);
      if (right != nai_v)
        split(right, indices, num_threads);
    }
  } else {
    std::deque<typename Node::ID> queue;
    queue.push_back(node_id);
    while (!queue.empty()) {
      typename Node::ID node_id = queue.front();
      queue.pop_front();

      std::tie(left, right) = ssplit(node_id, indices);
      if (left != nai_v)
        queue.push_back(left);
      if (right != nai_v)
        queue.push_back(right);
    }
  }
  *num_threads += 1;
}

template<uint16_t K, typename IdxType>
std::pair<typename KDTree<K, IdxType>::Node::ID, typename KDTree<K, IdxType>::Node::ID>
KDTree<K, IdxType>::ssplit(typename Node::ID node_id, std::vector<IdxType> *indices) {
  Node &node = nodes[node_id];
  decltype(K) d = node.d;
  std::sort(indices->data() + node.first, indices->data() + node.last,
            [this, d](IdxType a, IdxType b) -> bool {
              return vertices[a][d] < vertices[b][d];
            }
  );
  d = (d + 1) % K;
  IdxType mid = (node.last + node.first) / 2;
  node.vertex_id = indices->at(mid);
  if (mid - node.first > 0) {
    node.left = CreateNode(d, node.first, mid);
  }
  if (node.last - (mid + 1) > 0) {
    node.right = CreateNode(d, mid + 1, node.last);
  }
  return std::make_pair(node.left, node.right);
}

template<uint16_t K, typename IdxType>
std::pair<IdxType, double>
KDTree<K, IdxType>::find_nn(Eigen::Matrix<double, K, 1> point, double max_dist) const {
  return find_nns(point, 1, max_dist)[0];
}

template<uint16_t K, typename IdxType>
std::vector<std::pair<IdxType, double>>
KDTree<K, IdxType>::find_nns(Eigen::Matrix<double, K, 1> vertex, std::size_t n, double max_dist) const {

  std::pair<IdxType, double> nn = std::make_pair(nai_v, max_dist);
  std::vector<std::pair<IdxType, double>> nns(n, nn);

  std::stack<std::pair<typename Node::ID, bool> > s;
  s.emplace(0, true);
  while (!s.empty()) {
    typename Node::ID node_id;
    bool down;
    std::tie(node_id, down) = s.top();
    s.pop();

    if (node_id == nai_v)
      continue;

    Node const &node = nodes[node_id];

    double diff = vertex[node.d] - vertices[node.vertex_id][node.d];
    if (down) {
      double dist = (vertex - vertices[node.vertex_id]).norm();
      if (dist < max_dist) {
        nns.emplace_back(node.vertex_id, dist);
        std::sort(nns.begin(), nns.end(),
                  [](std::pair<IdxType, double> a, std::pair<IdxType, double> b) -> bool {
                    return a.second < b.second;
                  });
        nns.pop_back();
        max_dist = nns.back().second;
      }

      if (node.left == nai_v && node.right == nai_v)
        continue;

      s.emplace(node_id, false);
      if (diff < 0.0f) {
        s.emplace(node.left, true);
      } else {
        s.emplace(node.right, true);
      }
    } else {
      if (std::abs(diff) >= max_dist)
        continue;

      if (diff < 0.0f) {
        s.emplace(node.right, true);
      } else {
        s.emplace(node.left, true);
      }
    }
  }
  return nns;
}

}// namespace pcs