#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace std::chrono;

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

constexpr node_t DEPOT = 0;
constexpr int DEFAULT_ITERS = 100000;
constexpr int DEFAULT_BATCH = 4096;
constexpr int DEFAULT_BLOCK = 256;
constexpr int NUM_STREAMS = 2;

struct FlatMst {
  int n = 0;
  vector<int> nbr;
  vector<int> off;
  vector<int> deg;
};

struct Params {
  bool toRound = true;
  int nThreads = 20;
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
    size_t myoffset = ((2ULL * static_cast<size_t>(i) * size) - (static_cast<size_t>(i) * i) + i) / 2ULL;
    size_t correction = 2ULL * static_cast<size_t>(i) + 1ULL;
    return dist[myoffset + j - correction];
  }
};

struct GpuEvaluator {
  int* d_perms[NUM_STREAMS] = {nullptr, nullptr};
  double* d_costs[NUM_STREAMS] = {nullptr, nullptr};
  int* h_perms[NUM_STREAMS] = {nullptr, nullptr};
  double* h_costs[NUM_STREAMS] = {nullptr, nullptr};
  cudaStream_t streams[NUM_STREAMS] = {};
  double* d_dist = nullptr;
  double* d_demands = nullptr;
  size_t n = 0;
  size_t distSize = 0;
  int batchSize = 0;
};

static inline void cudaCheck(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    cerr << what << ": " << cudaGetErrorString(status) << '\n';
    exit(1);
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

vector<vector<Edge>> PrimsAlgo(const VRP& vrp, vector<vector<Edge>>& graph) {
  size_t n = graph.size();
  vector<weight_t> key(n, numeric_limits<weight_t>::max());
  vector<node_t> parent(n, -1);
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
    if (parent[v] < 0) {
      continue;
    }
    weight_t w = vrp.get_dist(v, parent[v]);
    tree[v].push_back(Edge(parent[v], w));
    tree[parent[v]].push_back(Edge(v, w));
  }

  return tree;
}

FlatMst buildFlatMst(const vector<vector<Edge>>& mst) {
  FlatMst fm;
  fm.n = static_cast<int>(mst.size());
  fm.off.resize(fm.n + 1);
  fm.deg.resize(fm.n);
  fm.off[0] = 0;
  for (int i = 0; i < fm.n; ++i) {
    fm.deg[i] = static_cast<int>(mst[i].size());
    fm.off[i + 1] = fm.off[i] + fm.deg[i];
  }
  fm.nbr.resize(fm.off[fm.n]);
  for (int i = 0; i < fm.n; ++i) {
    for (int j = 0; j < fm.deg[i]; ++j) {
      fm.nbr[fm.off[i] + j] = mst[i][j].to;
    }
  }
  return fm;
}

void generateBatchInto(const FlatMst& fm, int iterBase, int currentBatch, int nThreads, int* flatOut) {
  int n = fm.n;
#pragma omp parallel num_threads(nThreads)
  {
    vector<int> localNbr(fm.nbr.size());
    vector<bool> visited(n, false);
    vector<pair<int, int>> stk;
    stk.reserve(n);
    vector<int> perm;
    perm.reserve(n);
    mt19937 rng;

#pragma omp for schedule(static)
    for (int b = 0; b < currentBatch; ++b) {
      unsigned s = static_cast<unsigned>(iterBase + b);
      s ^= s >> 16;
      s *= 0x45d9f3bu;
      s ^= s >> 16;
      rng.seed(s);

      localNbr = fm.nbr;
      for (int i = 0; i < n; ++i) {
        if (fm.deg[i] > 1) {
          shuffle(localNbr.begin() + fm.off[i], localNbr.begin() + fm.off[i] + fm.deg[i], rng);
        }
      }

      perm.clear();
      fill(visited.begin(), visited.end(), false);
      stk.clear();
      visited[DEPOT] = true;
      perm.push_back(DEPOT);
      stk.push_back({DEPOT, 0});

      while (!stk.empty()) {
        int u = stk.back().first;
        int ci = stk.back().second;
        if (ci < fm.deg[u]) {
          stk.back().second++;
          int v = localNbr[fm.off[u] + ci];
          if (!visited[v]) {
            visited[v] = true;
            perm.push_back(v);
            stk.push_back({v, 0});
          }
        } else {
          stk.pop_back();
        }
      }

      int base = b * n;
      for (int i = 0; i < n; ++i) {
        flatOut[base + i] = perm[i];
      }
    }
  }
}

vector<vector<node_t>> convertToVrpRoutes(const VRP& vrp, const vector<node_t>& permutation) {
  vector<vector<node_t>> routes;
  vector<node_t> current;
  demand_t residue = vrp.getCapacity();

  for (node_t node : permutation) {
    if (node == DEPOT) {
      continue;
    }

    if (residue >= vrp.node[node].demand) {
      current.push_back(node);
      residue -= vrp.node[node].demand;
    } else {
      routes.push_back(current);
      current.clear();
      current.push_back(node);
      residue = vrp.getCapacity() - vrp.node[node].demand;
    }
  }

  routes.push_back(current);
  return routes;
}

weight_t get_total_cost_of_routes(const VRP& vrp, const vector<vector<node_t>>& routes) {
  weight_t total = 0.0;
  for (const auto& route : routes) {
    if (route.empty()) {
      continue;
    }
    weight_t routeCost = vrp.get_dist(DEPOT, route.front());
    for (size_t i = 1; i < route.size(); ++i) {
      routeCost += vrp.get_dist(route[i - 1], route[i]);
    }
    routeCost += vrp.get_dist(route.back(), DEPOT);
    total += routeCost;
  }
  return total;
}

void tsp_approx(const VRP& vrp, vector<node_t>& cities, vector<node_t>& tour, node_t cityCount) {
  for (node_t i = 1; i < cityCount; ++i) {
    tour[i] = cities[i - 1];
  }
  tour[0] = cities[cityCount - 1];

  for (node_t i = 1; i < cityCount; ++i) {
    weight_t baseX = vrp.node[tour[i - 1]].x;
    weight_t baseY = vrp.node[tour[i - 1]].y;
    weight_t best = DBL_MAX;
    node_t bestIdx = i;

    for (node_t j = cityCount - 1;; --j) {
      weight_t dx = vrp.node[tour[j]].x - baseX;
      weight_t dy = vrp.node[tour[j]].y - baseY;
      weight_t score = dx * dx + dy * dy;
      if (score <= best) {
        if (j < i) {
          break;
        }
        best = score;
        bestIdx = j;
      }
    }

    swap(tour[i], tour[bestIdx]);
  }
}

vector<vector<node_t>> postprocess_tsp_approx(const VRP& vrp, vector<vector<node_t>>& routes) {
  vector<vector<node_t>> out(routes.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
    size_t len = routes[i].size();
    vector<node_t> cities(len + 1);
    vector<node_t> tour(len + 1);
    for (size_t j = 0; j < len; ++j) {
      cities[j] = routes[i][j];
    }
    cities[len] = DEPOT;
    tsp_approx(vrp, cities, tour, static_cast<node_t>(len + 1));
    out[i].assign(tour.begin() + 1, tour.end());
  }

  return out;
}

void tsp_2opt(const VRP& vrp, vector<node_t>& cities, vector<node_t>& scratch, unsigned len) {
  unsigned improve = 0;

  while (improve < 2) {
    double best = vrp.get_dist(DEPOT, cities[0]);
    for (unsigned i = 1; i < len; ++i) {
      best += vrp.get_dist(cities[i - 1], cities[i]);
    }
    best += vrp.get_dist(DEPOT, cities[len - 1]);

    for (unsigned i = 0; i + 1 < len; ++i) {
      for (unsigned k = i + 1; k < len; ++k) {
        for (unsigned c = 0; c < i; ++c) {
          scratch[c] = cities[c];
        }
        for (unsigned c = i; c <= k; ++c) {
          scratch[c] = cities[k - (c - i)];
        }
        for (unsigned c = k + 1; c < len; ++c) {
          scratch[c] = cities[c];
        }

        double now = vrp.get_dist(DEPOT, scratch[0]);
        for (unsigned c = 1; c < len; ++c) {
          now += vrp.get_dist(scratch[c - 1], scratch[c]);
        }
        now += vrp.get_dist(DEPOT, scratch[len - 1]);

        if (now < best) {
          improve = 0;
          for (unsigned c = 0; c < len; ++c) {
            cities[c] = scratch[c];
          }
          best = now;
        }
      }
    }
    ++improve;
  }
}

vector<vector<node_t>> postprocess_2opt(const VRP& vrp, vector<vector<node_t>>& routes) {
  vector<vector<node_t>> out(routes.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
    vector<node_t> cities = routes[i];
    vector<node_t> scratch(cities.size());
    if (cities.size() > 2) {
      tsp_2opt(vrp, cities, scratch, static_cast<unsigned>(cities.size()));
    }
    out[i] = move(cities);
  }

  return out;
}

vector<vector<node_t>> postProcessIt(const VRP& vrp, vector<vector<node_t>>& routes, weight_t& minCost) {
  auto approxRoutes = postprocess_tsp_approx(vrp, routes);
  auto approxThen2Opt = postprocess_2opt(vrp, approxRoutes);
  auto direct2Opt = postprocess_2opt(vrp, routes);

  vector<vector<node_t>> out(routes.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(routes.size()); ++i) {
    vector<vector<node_t>> a{approxThen2Opt[i]};
    vector<vector<node_t>> b{direct2Opt[i]};
    out[i] = (get_total_cost_of_routes(vrp, a) <= get_total_cost_of_routes(vrp, b)) ? approxThen2Opt[i] : direct2Opt[i];
  }

  minCost = get_total_cost_of_routes(vrp, out);
  return out;
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

__global__ void evaluateBatchKernel(
    const int* perms,
    const double* dist,
    const double* demands,
    double* costs,
    int n,
    int batchSize,
    double capacity) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= batchSize) {
    return;
  }

  const int* perm = perms + static_cast<size_t>(tid) * n;
  double residue = capacity;
  double cost = 0.0;
  int prev = DEPOT;

  for (int i = 0; i < n; ++i) {
    int node = perm[i];
    if (node == DEPOT) {
      continue;
    }

    double demand = demands[node];
    if (residue >= demand) {
      cost += get_dist_device(dist, n, prev, node);
      residue -= demand;
      prev = node;
    } else {
      cost += get_dist_device(dist, n, prev, DEPOT);
      residue = capacity;
      prev = DEPOT;
      --i;
    }
  }

  cost += get_dist_device(dist, n, prev, DEPOT);
  costs[tid] = cost;
}

void initGpuEvaluator(GpuEvaluator& gpu, const VRP& vrp, int batchSize) {
  gpu.n = vrp.getSize();
  gpu.distSize = vrp.dist.size();
  gpu.batchSize = batchSize;

  cudaCheck(cudaMalloc(&gpu.d_dist, sizeof(double) * gpu.distSize), "cudaMalloc d_dist");
  cudaCheck(cudaMalloc(&gpu.d_demands, sizeof(double) * gpu.n), "cudaMalloc d_demands");
  for (int s = 0; s < NUM_STREAMS; ++s) {
    cudaCheck(cudaMalloc(&gpu.d_perms[s], sizeof(int) * gpu.n * batchSize), "cudaMalloc d_perms");
    cudaCheck(cudaMalloc(&gpu.d_costs[s], sizeof(double) * batchSize), "cudaMalloc d_costs");
    cudaCheck(cudaMallocHost(&gpu.h_perms[s], sizeof(int) * gpu.n * batchSize), "cudaMallocHost h_perms");
    cudaCheck(cudaMallocHost(&gpu.h_costs[s], sizeof(double) * batchSize), "cudaMallocHost h_costs");
    cudaCheck(cudaStreamCreate(&gpu.streams[s]), "cudaStreamCreate");
  }

  vector<double> demands(gpu.n);
  for (size_t i = 0; i < gpu.n; ++i) {
    demands[i] = vrp.node[i].demand;
  }

  cudaCheck(cudaMemcpy(gpu.d_dist, vrp.dist.data(), sizeof(double) * gpu.distSize, cudaMemcpyHostToDevice), "copy d_dist");
  cudaCheck(cudaMemcpy(gpu.d_demands, demands.data(), sizeof(double) * gpu.n, cudaMemcpyHostToDevice), "copy d_demands");
}

void cleanupGpuEvaluator(GpuEvaluator& gpu) {
  for (int s = 0; s < NUM_STREAMS; ++s) {
    cudaFree(gpu.d_perms[s]);
    cudaFree(gpu.d_costs[s]);
    if (gpu.h_perms[s]) {
      cudaFreeHost(gpu.h_perms[s]);
    }
    if (gpu.h_costs[s]) {
      cudaFreeHost(gpu.h_costs[s]);
    }
    cudaStreamDestroy(gpu.streams[s]);
  }
  cudaFree(gpu.d_dist);
  cudaFree(gpu.d_demands);
}

void launchGpuBatch(
    const VRP& vrp,
    GpuEvaluator& gpu,
    int slot,
    int batchSize,
    int blockSize) {
  cudaCheck(
      cudaMemcpyAsync(
          gpu.d_perms[slot],
          gpu.h_perms[slot],
          sizeof(int) * batchSize * vrp.getSize(),
          cudaMemcpyHostToDevice,
          gpu.streams[slot]),
      "copy perms async");
  int gridSize = (batchSize + blockSize - 1) / blockSize;
  evaluateBatchKernel<<<gridSize, blockSize>>>(
      gpu.d_perms[slot],
      gpu.d_dist,
      gpu.d_demands,
      gpu.d_costs[slot],
      static_cast<int>(vrp.getSize()),
      batchSize,
      vrp.getCapacity());
  cudaCheck(cudaGetLastError(), "evaluateBatchKernel launch");
  cudaCheck(
      cudaMemcpyAsync(
          gpu.h_costs[slot],
          gpu.d_costs[slot],
          sizeof(double) * batchSize,
          cudaMemcpyDeviceToHost,
          gpu.streams[slot]),
      "copy costs async");
}

vector<double> finishGpuBatch(GpuEvaluator& gpu, int slot, int batchSize) {
  cudaCheck(cudaStreamSynchronize(gpu.streams[slot]), "stream sync");
  return vector<double>(gpu.h_costs[slot], gpu.h_costs[slot] + batchSize);
}

int main(int argc, char* argv[]) {
  VRP vrp;
  if (argc < 2) {
    cout << "gpucpuMDS_v2\n";
    cout << "Usage: " << argv[0]
         << " toy.vrp [-nthreads N] [-round 0|1] [-iters N] [-batch N] [-block N]\n";
    return 1;
  }

  for (int i = 2; i < argc; i += 2) {
    string arg = argv[i];
    if (arg == "-round") {
      vrp.params.toRound = atoi(argv[i + 1]) != 0;
    } else if (arg == "-nthreads") {
      vrp.params.nThreads = max(1, atoi(argv[i + 1]));
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

  omp_set_num_threads(vrp.params.nThreads);
  vrp.read(argv[1]);

  auto start = high_resolution_clock::now();
  auto completeGraph = vrp.cal_graph_dist();
  auto mst = PrimsAlgo(vrp, completeGraph);
  auto fm = buildFlatMst(mst);

  GpuEvaluator gpu;
  initGpuEvaluator(gpu, vrp, vrp.params.batchSize);

  generateBatchInto(fm, 0, 1, vrp.params.nThreads, gpu.h_perms[0]);
  auto firstRoutes = convertToVrpRoutes(vrp, vector<node_t>(gpu.h_perms[0], gpu.h_perms[0] + fm.n));
  weight_t minCost = get_total_cost_of_routes(vrp, firstRoutes);
  vector<vector<node_t>> minRoutes = firstRoutes;
  weight_t minCost1 = minCost;

  auto end = high_resolution_clock::now();
  double timeUpto1 = duration_cast<duration<double>>(end - start).count();

  int n = fm.n;
  int nextIter = 1;
  int activeSlot = 0;
  int activeBatchSize = min(vrp.params.batchSize, vrp.params.totalIters - nextIter);
  generateBatchInto(fm, nextIter, activeBatchSize, vrp.params.nThreads, gpu.h_perms[activeSlot]);
  launchGpuBatch(vrp, gpu, activeSlot, activeBatchSize, vrp.params.blockSize);
  nextIter += activeBatchSize;

  while (activeBatchSize > 0) {
    int nextSlot = 1 - activeSlot;
    int nextBatchSize = 0;

    if (nextIter < vrp.params.totalIters) {
      nextBatchSize = min(vrp.params.batchSize, vrp.params.totalIters - nextIter);
      generateBatchInto(fm, nextIter, nextBatchSize, vrp.params.nThreads, gpu.h_perms[nextSlot]);
      launchGpuBatch(vrp, gpu, nextSlot, nextBatchSize, vrp.params.blockSize);
      nextIter += nextBatchSize;
    }

    auto costs = finishGpuBatch(gpu, activeSlot, activeBatchSize);
    for (int b = 0; b < activeBatchSize; ++b) {
      if (costs[b] < minCost) {
        minCost = costs[b];
        const int* p = gpu.h_perms[activeSlot] + b * n;
        minRoutes = convertToVrpRoutes(vrp, vector<node_t>(p, p + n));
      }
    }

    activeSlot = nextSlot;
    activeBatchSize = nextBatchSize;
  }

  weight_t minCost2 = minCost;
  end = high_resolution_clock::now();
  double timeUpto2 = duration_cast<duration<double>>(end - start).count();

  auto postRoutes = postProcessIt(vrp, minRoutes, minCost);

  end = high_resolution_clock::now();
  double totalTime = duration_cast<duration<double>>(end - start).count();

  bool verified = verify_sol(vrp, postRoutes, static_cast<unsigned>(vrp.getCapacity()));
  cerr << argv[1] << " Cost " << minCost1 << ' ' << minCost2 << ' ' << minCost;
  cerr << " Time(seconds) " << timeUpto1 << ' ' << timeUpto2 << ' ' << totalTime;
  cerr << " nthreads " << vrp.params.nThreads;
  cerr << " batch " << vrp.params.batchSize << " block " << vrp.params.blockSize;
  cerr << (verified ? " VALID\n" : " INVALID\n");

  printOutput(vrp, postRoutes);
  cleanupGpuEvaluator(gpu);
  return verified ? 0 : 2;
}
