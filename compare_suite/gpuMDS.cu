#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <limits>
#include <vector>

using namespace std;
using namespace std::chrono;

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

constexpr node_t DEPOT = 0;
constexpr int DEFAULT_ITERS = 100000;
constexpr int DEFAULT_BATCH = 256;
constexpr int DEFAULT_BLOCK = 128;

struct Params {
  bool toRound = true;
  int totalIters = DEFAULT_ITERS;
  int batchSize = DEFAULT_BATCH;
  int blockSize = DEFAULT_BLOCK;
};

struct Edge {
  node_t to = 0;
  weight_t length = 0.0;

  Edge() = default;
  Edge(node_t t, weight_t l) : to(t), length(l) {}
};

struct Point {
  point_t x = 0.0;
  point_t y = 0.0;
  demand_t demand = 0.0;
};

struct VRP {
  size_t size = 0;
  demand_t capacity = 0.0;
  string type;
  vector<Point> node;
  vector<weight_t> dist;
  Params params;

  unsigned read(const string& filename);
  void print() const;
  void print_dist() const;
  vector<vector<Edge>> cal_graph_dist();

  size_t getSize() const {
    return size;
  }

  demand_t getCapacity() const {
    return capacity;
  }

  weight_t get_dist(node_t i, node_t j) const {
    if (i == j) {
      return 0.0;
    }

    if (i > j) {
      swap(i, j);
    }

    size_t myoffset = ((2 * static_cast<size_t>(i) * size) - (static_cast<size_t>(i) * i) + i) / 2;
    size_t correction = 2 * static_cast<size_t>(i) + 1;
    return dist[myoffset + j - correction];
  }
};

struct CsrTree {
  vector<int> offsets;
  vector<int> neighbors;
};

struct DeviceBuffers {
  int* perms = nullptr;
  double* costs = nullptr;
  unsigned char* visited = nullptr;
  int* stacks = nullptr;
  int* offsets = nullptr;
  int* neighbors = nullptr;
  double* dist = nullptr;
  double* demands = nullptr;
};

struct FlatRoutes {
  vector<int> offsets;
  vector<int> nodes;
  int maxRouteLen = 0;
};

struct RouteDeviceBuffers {
  int* offsets = nullptr;
  int* inputNodes = nullptr;
  int* approxNodes = nullptr;
  int* approx2OptNodes = nullptr;
  int* direct2OptNodes = nullptr;
  int* finalNodes = nullptr;
  int* scratchA = nullptr;
  int* scratchB = nullptr;
  double* approxCosts = nullptr;
  double* directCosts = nullptr;
};

static inline void cudaCheck(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    cerr << what << ": " << cudaGetErrorString(status) << '\n';
    exit(1);
  }
}

vector<vector<Edge>> VRP::cal_graph_dist() {
  dist.resize((size * (size - 1)) / 2);
  vector<vector<Edge>> graph(size);

  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      weight_t w = sqrt(((node[i].x - node[j].x) * (node[i].x - node[j].x)) +
                        ((node[i].y - node[j].y) * (node[i].y - node[j].y)));

      dist[k] = params.toRound ? round(w) : w;
      graph[i].push_back(Edge(static_cast<node_t>(j), dist[k]));
      graph[j].push_back(Edge(static_cast<node_t>(i), dist[k]));
      ++k;
    }
  }

  return graph;
}

void VRP::print_dist() const {
  for (size_t i = 0; i < size; ++i) {
    cout << i << ':';
    for (size_t j = 0; j < size; ++j) {
      cout << setw(10) << get_dist(static_cast<node_t>(i), static_cast<node_t>(j)) << ' ';
    }
    cout << '\n';
  }
}

unsigned VRP::read(const string& filename) {
  ifstream in(filename);
  if (!in.is_open()) {
    cerr << "Could not open the file \"" << filename << "\"\n";
    exit(1);
  }

  string line;
  for (int i = 0; i < 3; ++i) {
    getline(in, line);
  }

  getline(in, line);
  size = static_cast<size_t>(stof(line.substr(line.find(':') + 2)));

  getline(in, line);
  type = line.substr(line.find(':') + 2);

  getline(in, line);
  capacity = stof(line.substr(line.find(':') + 2));

  getline(in, line);
  node.resize(size);

  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    stringstream iss(line);
    size_t id;
    string xStr;
    string yStr;
    iss >> id >> xStr >> yStr;
    node[i].x = stof(xStr);
    node[i].y = stof(yStr);
  }

  getline(in, line);

  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    stringstream iss(line);
    size_t id;
    string dStr;
    iss >> id >> dStr;
    node[i].demand = stof(dStr);
  }

  return static_cast<unsigned>(capacity);
}

void VRP::print() const {
  cout << "DIMENSION:" << size << '\n';
  cout << "CAPACITY:" << capacity << '\n';
  for (size_t i = 0; i < size; ++i) {
    cout << i << ':'
         << setw(6) << node[i].x << ' '
         << setw(6) << node[i].y << ' '
         << setw(6) << node[i].demand << '\n';
  }
}

vector<vector<Edge>> PrimsAlgo(const VRP& vrp, vector<vector<Edge>>& graph) {
  const node_t init = -1;
  size_t n = graph.size();

  vector<weight_t> key(n, numeric_limits<weight_t>::max());
  vector<node_t> parent(n, init);
  vector<bool> visited(n, false);
  set<pair<weight_t, node_t>> active;
  vector<vector<Edge>> tree(n);

  key[DEPOT] = 0.0;
  active.insert({0.0, DEPOT});

  while (!active.empty()) {
    node_t u = active.begin()->second;
    active.erase(active.begin());
    if (visited[u]) {
      continue;
    }

    visited[u] = true;
    for (const Edge& edge : graph[u]) {
      if (!visited[edge.to] && edge.length < key[edge.to]) {
        key[edge.to] = edge.length;
        parent[edge.to] = u;
        active.insert({key[edge.to], edge.to});
      }
    }
  }

  for (node_t v = 0; v < static_cast<node_t>(n); ++v) {
    if (parent[v] == init) {
      continue;
    }

    weight_t w = vrp.get_dist(v, parent[v]);
    tree[v].push_back(Edge(parent[v], w));
    tree[parent[v]].push_back(Edge(v, w));
  }

  return tree;
}

CsrTree buildCsrTree(const vector<vector<Edge>>& tree) {
  CsrTree csr;
  csr.offsets.resize(tree.size() + 1, 0);

  size_t totalEdges = 0;
  for (size_t i = 0; i < tree.size(); ++i) {
    csr.offsets[i] = static_cast<int>(totalEdges);
    totalEdges += tree[i].size();
  }
  csr.offsets[tree.size()] = static_cast<int>(totalEdges);
  csr.neighbors.resize(totalEdges);

  size_t cursor = 0;
  for (const auto& edges : tree) {
    for (const Edge& edge : edges) {
      csr.neighbors[cursor++] = edge.to;
    }
  }

  return csr;
}

vector<vector<node_t>> convertToVrpRoutes(const VRP& vrp, const vector<node_t>& permutation) {
  vector<vector<node_t>> routes;
  vector<node_t> currentRoute;
  demand_t residue = vrp.getCapacity();

  for (node_t node : permutation) {
    if (node == DEPOT) {
      continue;
    }

    if (residue >= vrp.node[node].demand) {
      currentRoute.push_back(node);
      residue -= vrp.node[node].demand;
      continue;
    }

    routes.push_back(currentRoute);
    currentRoute.clear();
    currentRoute.push_back(node);
    residue = vrp.getCapacity() - vrp.node[node].demand;
  }

  routes.push_back(currentRoute);
  return routes;
}

FlatRoutes flattenRoutes(const vector<vector<node_t>>& routes) {
  FlatRoutes flat;
  flat.offsets.resize(routes.size() + 1, 0);

  int cursor = 0;
  for (size_t i = 0; i < routes.size(); ++i) {
    flat.offsets[i] = cursor;
    flat.maxRouteLen = max(flat.maxRouteLen, static_cast<int>(routes[i].size()));
    cursor += static_cast<int>(routes[i].size());
  }
  flat.offsets[routes.size()] = cursor;
  flat.nodes.resize(cursor);

  cursor = 0;
  for (const auto& route : routes) {
    for (node_t node : route) {
      flat.nodes[cursor++] = node;
    }
  }

  return flat;
}

vector<vector<node_t>> expandRoutes(const FlatRoutes& flat, const vector<int>& nodes) {
  vector<vector<node_t>> routes(flat.offsets.size() - 1);
  for (size_t i = 0; i + 1 < flat.offsets.size(); ++i) {
    int begin = flat.offsets[i];
    int end = flat.offsets[i + 1];
    routes[i].assign(nodes.begin() + begin, nodes.begin() + end);
  }
  return routes;
}

weight_t get_total_cost_of_routes(const VRP& vrp, const vector<vector<node_t>>& routes) {
  weight_t total = 0.0;

  for (const auto& route : routes) {
    if (route.empty()) {
      continue;
    }

    weight_t cost = vrp.get_dist(DEPOT, route.front());
    for (size_t i = 1; i < route.size(); ++i) {
      cost += vrp.get_dist(route[i - 1], route[i]);
    }
    cost += vrp.get_dist(route.back(), DEPOT);
    total += cost;
  }

  return total;
}

void printOutput(const VRP& vrp, const vector<vector<node_t>>& routes) {
  for (size_t i = 0; i < routes.size(); ++i) {
    cout << "Route #" << i + 1 << ':';
    for (node_t node : routes[i]) {
      cout << ' ' << node;
    }
    cout << '\n';
  }

  cout << "Cost " << get_total_cost_of_routes(vrp, routes) << '\n';
}

void tsp_approx(const VRP& vrp, vector<node_t>& cities, vector<node_t>& tour, node_t cityCount) {
  for (node_t i = 1; i < cityCount; ++i) {
    tour[i] = cities[i - 1];
  }
  tour[0] = cities[cityCount - 1];

  for (node_t i = 1; i < cityCount; ++i) {
    weight_t thisX = vrp.node[tour[i - 1]].x;
    weight_t thisY = vrp.node[tour[i - 1]].y;
    weight_t closest = DBL_MAX;
    node_t closeIdx = i;

    for (node_t j = cityCount - 1;; --j) {
      weight_t distX = vrp.node[tour[j]].x - thisX;
      weight_t distY = vrp.node[tour[j]].y - thisY;
      weight_t squared = distX * distX + distY * distY;
      if (squared <= closest) {
        if (j < i) {
          break;
        }
        closest = squared;
        closeIdx = j;
      }
    }

    swap(tour[i], tour[closeIdx]);
  }
}

vector<vector<node_t>> postprocess_tsp_approx(const VRP& vrp, vector<vector<node_t>>& routes) {
  vector<vector<node_t>> output;
  output.reserve(routes.size());

  for (const auto& route : routes) {
    size_t size = route.size();
    vector<node_t> cities(size + 1);
    vector<node_t> tour(size + 1);

    for (size_t i = 0; i < size; ++i) {
      cities[i] = route[i];
    }
    cities[size] = DEPOT;

    tsp_approx(vrp, cities, tour, static_cast<node_t>(size + 1));
    output.emplace_back(tour.begin() + 1, tour.end());
  }

  return output;
}

void tsp_2opt(const VRP& vrp, vector<node_t>& cities, vector<node_t>& scratch, unsigned cityCount) {
  unsigned improve = 0;

  while (improve < 2) {
    double bestDistance = vrp.get_dist(DEPOT, cities[0]);
    for (unsigned i = 1; i < cityCount; ++i) {
      bestDistance += vrp.get_dist(cities[i - 1], cities[i]);
    }
    bestDistance += vrp.get_dist(DEPOT, cities[cityCount - 1]);

    for (unsigned i = 0; i + 1 < cityCount; ++i) {
      for (unsigned k = i + 1; k < cityCount; ++k) {
        for (unsigned c = 0; c < i; ++c) {
          scratch[c] = cities[c];
        }

        for (unsigned c = i; c <= k; ++c) {
          scratch[c] = cities[k - (c - i)];
        }

        for (unsigned c = k + 1; c < cityCount; ++c) {
          scratch[c] = cities[c];
        }

        double newDistance = vrp.get_dist(DEPOT, scratch[0]);
        for (unsigned c = 1; c < cityCount; ++c) {
          newDistance += vrp.get_dist(scratch[c - 1], scratch[c]);
        }
        newDistance += vrp.get_dist(DEPOT, scratch[cityCount - 1]);

        if (newDistance < bestDistance) {
          improve = 0;
          for (unsigned c = 0; c < cityCount; ++c) {
            cities[c] = scratch[c];
          }
          bestDistance = newDistance;
        }
      }
    }

    ++improve;
  }
}

vector<vector<node_t>> postprocess_2opt(const VRP& vrp, vector<vector<node_t>>& routes) {
  vector<vector<node_t>> output;
  output.reserve(routes.size());

  for (const auto& route : routes) {
    vector<node_t> cities(route.begin(), route.end());
    vector<node_t> scratch(route.size());
    if (cities.size() > 2) {
      tsp_2opt(vrp, cities, scratch, static_cast<unsigned>(cities.size()));
    }
    output.push_back(cities);
  }

  return output;
}

vector<vector<node_t>> postProcessIt(const VRP& vrp, vector<vector<node_t>>& routes, weight_t& minCost) {
  auto approxRoutes = postprocess_tsp_approx(vrp, routes);
  auto approxThen2Opt = postprocess_2opt(vrp, approxRoutes);
  auto direct2Opt = postprocess_2opt(vrp, routes);

  vector<vector<node_t>> output(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) {
    vector<vector<node_t>> route2{approxThen2Opt[i]};
    vector<vector<node_t>> route3{direct2Opt[i]};
    if (get_total_cost_of_routes(vrp, route2) <= get_total_cost_of_routes(vrp, route3)) {
      output[i] = approxThen2Opt[i];
    } else {
      output[i] = direct2Opt[i];
    }
  }

  minCost = get_total_cost_of_routes(vrp, output);
  return output;
}

bool verify_sol(const VRP& vrp, const vector<vector<node_t>>& routes, unsigned capacity) {
  vector<unsigned> seen(vrp.getSize(), 0);

  for (const auto& route : routes) {
    unsigned load = 0;
    for (node_t node : route) {
      load += static_cast<unsigned>(vrp.node[node].demand);
      ++seen[node];
    }
    if (load > capacity) {
      return false;
    }
  }

  for (size_t i = 1; i < vrp.getSize(); ++i) {
    if (seen[i] != 1) {
      return false;
    }
  }

  return true;
}

__device__ unsigned mix32(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ double get_dist_device(const double* dist, int n, int i, int j) {
  if (i == j) {
    return 0.0;
  }

  if (i > j) {
    int tmp = i;
    i = j;
    j = tmp;
  }

  size_t myoffset = ((2ULL * static_cast<size_t>(i) * n) - (static_cast<size_t>(i) * i) + i) / 2ULL;
  size_t correction = 2ULL * static_cast<size_t>(i) + 1ULL;
  return dist[myoffset + j - correction];
}

__device__ double route_cost_device(const int* offsets, const int* nodes, const double* dist, int n, int routeIdx) {
  int begin = offsets[routeIdx];
  int end = offsets[routeIdx + 1];
  if (begin >= end) {
    return 0.0;
  }

  double cost = get_dist_device(dist, n, DEPOT, nodes[begin]);
  for (int i = begin + 1; i < end; ++i) {
    cost += get_dist_device(dist, n, nodes[i - 1], nodes[i]);
  }
  cost += get_dist_device(dist, n, nodes[end - 1], DEPOT);
  return cost;
}

__global__ void nearestNeighborRoutesKernel(
    const int* offsets,
    const int* inputNodes,
    int* outputNodes,
    unsigned char* visited,
    const double* dist,
    int n,
    int routeCount,
    int maxRouteLen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= routeCount) {
    return;
  }

  int begin = offsets[tid];
  int end = offsets[tid + 1];
  int len = end - begin;
  if (len <= 0) {
    return;
  }

  unsigned char* myVisited = visited + static_cast<size_t>(tid) * maxRouteLen;
  for (int i = 0; i < len; ++i) {
    myVisited[i] = 0;
  }

  int prev = DEPOT;
  for (int pos = 0; pos < len; ++pos) {
    int bestLocal = -1;
    double bestDist = DBL_MAX;

    for (int j = 0; j < len; ++j) {
      if (myVisited[j]) {
        continue;
      }
      int node = inputNodes[begin + j];
      double candidate = get_dist_device(dist, n, prev, node);
      if (candidate < bestDist) {
        bestDist = candidate;
        bestLocal = j;
      }
    }

    myVisited[bestLocal] = 1;
    outputNodes[begin + pos] = inputNodes[begin + bestLocal];
    prev = outputNodes[begin + pos];
  }
}

__global__ void twoOptRoutesKernel(
    const int* offsets,
    const int* inputNodes,
    int* outputNodes,
    int* scratchA,
    int* scratchB,
    const double* dist,
    int n,
    int routeCount,
    int maxRouteLen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= routeCount) {
    return;
  }

  int begin = offsets[tid];
  int end = offsets[tid + 1];
  int len = end - begin;
  int* cities = scratchA + static_cast<size_t>(tid) * maxRouteLen;
  int* tour = scratchB + static_cast<size_t>(tid) * maxRouteLen;

  for (int i = 0; i < len; ++i) {
    cities[i] = inputNodes[begin + i];
  }

  if (len > 2) {
    int improve = 0;
    while (improve < 2) {
      double bestDistance = get_dist_device(dist, n, DEPOT, cities[0]);
      for (int i = 1; i < len; ++i) {
        bestDistance += get_dist_device(dist, n, cities[i - 1], cities[i]);
      }
      bestDistance += get_dist_device(dist, n, DEPOT, cities[len - 1]);

      for (int i = 0; i + 1 < len; ++i) {
        for (int k = i + 1; k < len; ++k) {
          for (int c = 0; c < i; ++c) {
            tour[c] = cities[c];
          }
          for (int c = i; c <= k; ++c) {
            tour[c] = cities[k - (c - i)];
          }
          for (int c = k + 1; c < len; ++c) {
            tour[c] = cities[c];
          }

          double newDistance = get_dist_device(dist, n, DEPOT, tour[0]);
          for (int c = 1; c < len; ++c) {
            newDistance += get_dist_device(dist, n, tour[c - 1], tour[c]);
          }
          newDistance += get_dist_device(dist, n, DEPOT, tour[len - 1]);

          if (newDistance < bestDistance) {
            improve = 0;
            for (int c = 0; c < len; ++c) {
              cities[c] = tour[c];
            }
            bestDistance = newDistance;
          }
        }
      }
      ++improve;
    }
  }

  for (int i = 0; i < len; ++i) {
    outputNodes[begin + i] = cities[i];
  }
}

__global__ void routeCostKernel(
    const int* offsets,
    const int* nodes,
    double* costs,
    const double* dist,
    int n,
    int routeCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= routeCount) {
    return;
  }

  costs[tid] = route_cost_device(offsets, nodes, dist, n, tid);
}

__global__ void selectBestRoutesKernel(
    const int* offsets,
    const int* approxNodes,
    const int* directNodes,
    const double* approxCosts,
    const double* directCosts,
    int* finalNodes,
    int routeCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= routeCount) {
    return;
  }

  int begin = offsets[tid];
  int end = offsets[tid + 1];
  const int* src = (approxCosts[tid] <= directCosts[tid]) ? approxNodes : directNodes;
  for (int i = begin; i < end; ++i) {
    finalNodes[i] = src[i];
  }
}

__global__ void generateAndEvaluateKernel(
    int* perms,
    double* costs,
    unsigned char* visited,
    int* stacks,
    const int* offsets,
    const int* neighbors,
    const double* dist,
    const double* demands,
    int n,
    int batchSize,
    double capacity,
    unsigned seedBase) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= batchSize) {
    return;
  }

  int* myPerm = perms + static_cast<size_t>(tid) * n;
  unsigned char* myVisited = visited + static_cast<size_t>(tid) * n;
  int* myStack = stacks + static_cast<size_t>(tid) * n;

  for (int i = 0; i < n; ++i) {
    myVisited[i] = 0;
  }

  myStack[0] = DEPOT;
  int top = 0;
  int permLen = 0;

  while (top >= 0) {
    int u = myStack[top];
    if (!myVisited[u]) {
      myVisited[u] = 1;
      myPerm[permLen++] = u;
    }

    int bestV = -1;
    unsigned bestPriority = UINT_MAX;
    int begin = offsets[u];
    int end = offsets[u + 1];

    for (int idx = begin; idx < end; ++idx) {
      int v = neighbors[idx];
      if (myVisited[v]) {
        continue;
      }

      unsigned priority = mix32(seedBase ^ mix32(static_cast<unsigned>(tid + 1) * 2654435761U) ^
                                mix32(static_cast<unsigned>(u + 1) * 374761393U) ^
                                mix32(static_cast<unsigned>(v + 1) * 668265263U) ^
                                static_cast<unsigned>(idx));

      if (priority < bestPriority) {
        bestPriority = priority;
        bestV = v;
      }
    }

    if (bestV >= 0) {
      myStack[++top] = bestV;
    } else {
      --top;
    }
  }

  double cost = 0.0;
  double residue = capacity;
  int prev = DEPOT;
  int idx = 0;

  while (idx < permLen) {
    int node = myPerm[idx];
    if (node == DEPOT) {
      ++idx;
      continue;
    }

    double demand = demands[node];
    if (residue >= demand) {
      cost += get_dist_device(dist, n, prev, node);
      residue -= demand;
      prev = node;
      ++idx;
    } else {
      cost += get_dist_device(dist, n, prev, DEPOT);
      residue = capacity;
      prev = DEPOT;
    }
  }

  cost += get_dist_device(dist, n, prev, DEPOT);
  costs[tid] = cost;
}

void initDeviceBuffers(DeviceBuffers& buffers, const VRP& vrp, const CsrTree& csr, int batchSize) {
  size_t n = vrp.getSize();
  size_t distBytes = sizeof(double) * vrp.dist.size();
  size_t demandBytes = sizeof(double) * n;
  size_t offsetBytes = sizeof(int) * csr.offsets.size();
  size_t neighborBytes = sizeof(int) * csr.neighbors.size();

  cudaCheck(cudaMalloc(&buffers.perms, sizeof(int) * batchSize * n), "cudaMalloc perms");
  cudaCheck(cudaMalloc(&buffers.costs, sizeof(double) * batchSize), "cudaMalloc costs");
  cudaCheck(cudaMalloc(&buffers.visited, sizeof(unsigned char) * batchSize * n), "cudaMalloc visited");
  cudaCheck(cudaMalloc(&buffers.stacks, sizeof(int) * batchSize * n), "cudaMalloc stacks");
  cudaCheck(cudaMalloc(&buffers.offsets, offsetBytes), "cudaMalloc offsets");
  cudaCheck(cudaMalloc(&buffers.neighbors, neighborBytes), "cudaMalloc neighbors");
  cudaCheck(cudaMalloc(&buffers.dist, distBytes), "cudaMalloc dist");
  cudaCheck(cudaMalloc(&buffers.demands, demandBytes), "cudaMalloc demands");

  vector<double> demands(n);
  for (size_t i = 0; i < n; ++i) {
    demands[i] = vrp.node[i].demand;
  }

  cudaCheck(cudaMemcpy(buffers.offsets, csr.offsets.data(), offsetBytes, cudaMemcpyHostToDevice), "copy offsets");
  cudaCheck(cudaMemcpy(buffers.neighbors, csr.neighbors.data(), neighborBytes, cudaMemcpyHostToDevice), "copy neighbors");
  cudaCheck(cudaMemcpy(buffers.dist, vrp.dist.data(), distBytes, cudaMemcpyHostToDevice), "copy dist");
  cudaCheck(cudaMemcpy(buffers.demands, demands.data(), demandBytes, cudaMemcpyHostToDevice), "copy demands");
}

void cleanupDeviceBuffers(DeviceBuffers& buffers) {
  cudaFree(buffers.perms);
  cudaFree(buffers.costs);
  cudaFree(buffers.visited);
  cudaFree(buffers.stacks);
  cudaFree(buffers.offsets);
  cudaFree(buffers.neighbors);
  cudaFree(buffers.dist);
  cudaFree(buffers.demands);
}

void initRouteDeviceBuffers(RouteDeviceBuffers& buffers, const FlatRoutes& routes) {
  size_t offsetBytes = sizeof(int) * routes.offsets.size();
  size_t nodeBytes = sizeof(int) * routes.nodes.size();
  size_t scratchBytes = sizeof(int) * routes.offsets.size() * max(1, routes.maxRouteLen);
  size_t costBytes = sizeof(double) * (routes.offsets.size() - 1);

  cudaCheck(cudaMalloc(&buffers.offsets, offsetBytes), "cudaMalloc route offsets");
  cudaCheck(cudaMalloc(&buffers.inputNodes, nodeBytes), "cudaMalloc route input");
  cudaCheck(cudaMalloc(&buffers.approxNodes, nodeBytes), "cudaMalloc route approx");
  cudaCheck(cudaMalloc(&buffers.approx2OptNodes, nodeBytes), "cudaMalloc route approx2opt");
  cudaCheck(cudaMalloc(&buffers.direct2OptNodes, nodeBytes), "cudaMalloc route direct2opt");
  cudaCheck(cudaMalloc(&buffers.finalNodes, nodeBytes), "cudaMalloc route final");
  cudaCheck(cudaMalloc(&buffers.scratchA, scratchBytes), "cudaMalloc route scratchA");
  cudaCheck(cudaMalloc(&buffers.scratchB, scratchBytes), "cudaMalloc route scratchB");
  cudaCheck(cudaMalloc(&buffers.approxCosts, costBytes), "cudaMalloc route approx costs");
  cudaCheck(cudaMalloc(&buffers.directCosts, costBytes), "cudaMalloc route direct costs");

  cudaCheck(cudaMemcpy(buffers.offsets, routes.offsets.data(), offsetBytes, cudaMemcpyHostToDevice), "copy route offsets");
  cudaCheck(cudaMemcpy(buffers.inputNodes, routes.nodes.data(), nodeBytes, cudaMemcpyHostToDevice), "copy route nodes");
}

void cleanupRouteDeviceBuffers(RouteDeviceBuffers& buffers) {
  cudaFree(buffers.offsets);
  cudaFree(buffers.inputNodes);
  cudaFree(buffers.approxNodes);
  cudaFree(buffers.approx2OptNodes);
  cudaFree(buffers.direct2OptNodes);
  cudaFree(buffers.finalNodes);
  cudaFree(buffers.scratchA);
  cudaFree(buffers.scratchB);
  cudaFree(buffers.approxCosts);
  cudaFree(buffers.directCosts);
}

vector<vector<node_t>> postProcessGpu(
    const VRP& vrp,
    const DeviceBuffers& deviceBuffers,
    const vector<vector<node_t>>& routes,
    weight_t& minCost) {
  FlatRoutes flat = flattenRoutes(routes);
  if (flat.offsets.size() <= 1 || flat.nodes.empty()) {
    minCost = 0.0;
    return routes;
  }

  RouteDeviceBuffers buffers;
  initRouteDeviceBuffers(buffers, flat);

  int routeCount = static_cast<int>(routes.size());
  int blockSize = 128;
  int gridSize = (routeCount + blockSize - 1) / blockSize;

  unsigned char* visited = nullptr;
  cudaCheck(cudaMalloc(&visited, static_cast<size_t>(routeCount) * max(1, flat.maxRouteLen)), "cudaMalloc nn visited");

  nearestNeighborRoutesKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.inputNodes,
      buffers.approxNodes,
      visited,
      deviceBuffers.dist,
      static_cast<int>(vrp.getSize()),
      routeCount,
      max(1, flat.maxRouteLen));
  cudaCheck(cudaGetLastError(), "nearest kernel launch");

  twoOptRoutesKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.approxNodes,
      buffers.approx2OptNodes,
      buffers.scratchA,
      buffers.scratchB,
      deviceBuffers.dist,
      static_cast<int>(vrp.getSize()),
      routeCount,
      max(1, flat.maxRouteLen));
  cudaCheck(cudaGetLastError(), "approx 2opt kernel launch");

  twoOptRoutesKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.inputNodes,
      buffers.direct2OptNodes,
      buffers.scratchA,
      buffers.scratchB,
      deviceBuffers.dist,
      static_cast<int>(vrp.getSize()),
      routeCount,
      max(1, flat.maxRouteLen));
  cudaCheck(cudaGetLastError(), "direct 2opt kernel launch");

  routeCostKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.approx2OptNodes,
      buffers.approxCosts,
      deviceBuffers.dist,
      static_cast<int>(vrp.getSize()),
      routeCount);
  cudaCheck(cudaGetLastError(), "approx cost kernel launch");

  routeCostKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.direct2OptNodes,
      buffers.directCosts,
      deviceBuffers.dist,
      static_cast<int>(vrp.getSize()),
      routeCount);
  cudaCheck(cudaGetLastError(), "direct cost kernel launch");

  selectBestRoutesKernel<<<gridSize, blockSize>>>(
      buffers.offsets,
      buffers.approx2OptNodes,
      buffers.direct2OptNodes,
      buffers.approxCosts,
      buffers.directCosts,
      buffers.finalNodes,
      routeCount);
  cudaCheck(cudaGetLastError(), "select kernel launch");
  cudaCheck(cudaDeviceSynchronize(), "postprocess kernel sync");

  vector<int> finalNodes(flat.nodes.size());
  cudaCheck(cudaMemcpy(finalNodes.data(), buffers.finalNodes, sizeof(int) * finalNodes.size(), cudaMemcpyDeviceToHost), "copy final routes");
  cudaFree(visited);
  cleanupRouteDeviceBuffers(buffers);

  auto output = expandRoutes(flat, finalNodes);
  minCost = get_total_cost_of_routes(vrp, output);
  return output;
}

pair<double, vector<int>> runGpuBatch(
    const VRP& vrp,
    const DeviceBuffers& buffers,
    int currentBatch,
    unsigned seedBase,
    int blockSize) {
  int gridSize = (currentBatch + blockSize - 1) / blockSize;
  generateAndEvaluateKernel<<<gridSize, blockSize>>>(
      buffers.perms,
      buffers.costs,
      buffers.visited,
      buffers.stacks,
      buffers.offsets,
      buffers.neighbors,
      buffers.dist,
      buffers.demands,
      static_cast<int>(vrp.getSize()),
      currentBatch,
      vrp.getCapacity(),
      seedBase);

  cudaCheck(cudaGetLastError(), "kernel launch");
  cudaCheck(cudaDeviceSynchronize(), "kernel sync");

  vector<double> costs(currentBatch);
  cudaCheck(cudaMemcpy(costs.data(), buffers.costs, sizeof(double) * currentBatch, cudaMemcpyDeviceToHost), "copy costs");

  int bestIndex = 0;
  for (int i = 1; i < currentBatch; ++i) {
    if (costs[i] < costs[bestIndex]) {
      bestIndex = i;
    }
  }

  vector<int> bestPerm(vrp.getSize());
  cudaCheck(
      cudaMemcpy(
          bestPerm.data(),
          buffers.perms + static_cast<size_t>(bestIndex) * vrp.getSize(),
          sizeof(int) * vrp.getSize(),
          cudaMemcpyDeviceToHost),
      "copy best permutation");

  return {costs[bestIndex], move(bestPerm)};
}

int main(int argc, char* argv[]) {
  VRP vrp;
  if (argc < 2) {
    cout << "gpuMDS\n";
    cout << "Usage: " << argv[0]
         << " toy.vrp [-round 0|1] [-iters N] [-batch N] [-block N]\n";
    return 1;
  }

  for (int i = 2; i < argc; i += 2) {
    string arg = argv[i];
    if (arg == "-round") {
      vrp.params.toRound = atoi(argv[i + 1]) != 0;
    } else if (arg == "-iters") {
      vrp.params.totalIters = max(1, atoi(argv[i + 1]));
    } else if (arg == "-batch") {
      vrp.params.batchSize = max(1, atoi(argv[i + 1]));
    } else if (arg == "-block") {
      vrp.params.blockSize = max(1, atoi(argv[i + 1]));
    } else {
      cerr << "Invalid argument: " << arg << '\n';
      return 1;
    }
  }

  vrp.read(argv[1]);

  auto start = high_resolution_clock::now();
  auto completeGraph = vrp.cal_graph_dist();
  auto mst = PrimsAlgo(vrp, completeGraph);
  auto csr = buildCsrTree(mst);

  DeviceBuffers buffers;
  initDeviceBuffers(buffers, vrp, csr, vrp.params.batchSize);

  auto firstResult = runGpuBatch(vrp, buffers, 1, 0U, vrp.params.blockSize);
  weight_t minCost = firstResult.first;
  vector<int> bestPermutation = firstResult.second;
  weight_t minCost1 = minCost;

  auto end = high_resolution_clock::now();
  double timeUpto1 = duration_cast<duration<double>>(end - start).count();

  int done = 1;
  while (done < vrp.params.totalIters) {
    int currentBatch = min(vrp.params.batchSize, vrp.params.totalIters - done);
    auto batchResult = runGpuBatch(vrp, buffers, currentBatch, static_cast<unsigned>(done), vrp.params.blockSize);
    if (batchResult.first < minCost) {
      minCost = batchResult.first;
      bestPermutation = move(batchResult.second);
    }
    done += currentBatch;
  }

  weight_t minCost2 = minCost;
  end = high_resolution_clock::now();
  double timeUpto2 = duration_cast<duration<double>>(end - start).count();

  auto routes = convertToVrpRoutes(vrp, bestPermutation);
  auto postRoutes = postProcessGpu(vrp, buffers, routes, minCost);

  end = high_resolution_clock::now();
  double totalTime = duration_cast<duration<double>>(end - start).count();

  bool verified = verify_sol(vrp, postRoutes, static_cast<unsigned>(vrp.getCapacity()));

  cerr << argv[1] << " Cost " << minCost1 << ' ' << minCost2 << ' ' << minCost;
  cerr << " Time(seconds) " << timeUpto1 << ' ' << timeUpto2 << ' ' << totalTime;
  cerr << " batch " << vrp.params.batchSize << " block " << vrp.params.blockSize;
  cerr << (verified ? " VALID\n" : " INVALID\n");

  printOutput(vrp, postRoutes);
  cleanupDeviceBuffers(buffers);
  return verified ? 0 : 2;
}
