
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <random>
#include <chrono>

unsigned DEBUGCODE = 0;
#define DEBUG if (DEBUGCODE)

using namespace std;

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

const node_t DEPOT = 0;

class Params {
  public:
  Params() {
    toRound = 1;
    nThreads = 20;
  }
  ~Params() {}

  bool toRound;
  short nThreads;
};

class Edge {
  public:
  node_t to;
  weight_t length;

  Edge() {}
  ~Edge() {}
  Edge(node_t t, weight_t l) {
    to = t;
    length = l;
  }
  bool operator<(const Edge &e) {
    return length < e.length;
  }
};

class Point {
  public:

  point_t x;
  point_t y;
  demand_t demand;
};

class VRP {
  size_t size;
  demand_t capacity;
  string type;

  public:
  VRP() {}
  ~VRP() {}
  unsigned read(string filename);
  void print();

  void print_dist();

  std::vector<std::vector<Edge>> cal_graph_dist();
  weight_t get_dist(node_t i, node_t j) const {
    if (i == j)
      return 0.0;
    node_t temp;
    if (i > j) {
      temp = i;
      i = j;
      j = temp;
    }

    size_t myoffset = ((2 * i * size) - (i * i) + i) / 2;
    size_t correction = 2 * i + 1;
    return dist[myoffset + j - correction];
  }

  public:
  vector<Point> node;
  vector<weight_t> dist;
  Params params;

  size_t getSize() const {
    return size;
  }
  demand_t getCapacity() const {
    return capacity;
  }
};

std::vector<std::vector<Edge>>
VRP::cal_graph_dist() {

  dist.resize((size * (size - 1)) / 2);

  std::vector<std::vector<Edge>> nG(size);

  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      weight_t w = sqrt(((node[i].x - node[j].x) * (node[i].x - node[j].x)) + ((node[i].y - node[j].y) * (node[i].y - node[j].y)));

      dist[k] = (params.toRound ? round(w) : w);

      nG[i].push_back(Edge(j, w));
      nG[j].push_back(Edge(i, w));

      k++;
    }
  }

  return nG;
}

void VRP::print_dist() {
  for (size_t i = 0; i < size; ++i) {
    std::cout << i << ":";
    for (size_t j = 0; j < size; ++j) {
      cout << setw(10) << get_dist(i, j) << ' ';
    }
    std::cout << std::endl;
  }
}

unsigned VRP::read(string filename) {
  ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Could not open the file \"" << filename << "\"" << std::endl;
    exit(1);
  }
  string line;
  for (int i = 0; i < 3; ++i)
    getline(in, line);

  getline(in, line);
  size = stof(line.substr(line.find(":") + 2));

  getline(in, line);
  type = line.find(":");

  getline(in, line);
  capacity = stof(line.substr(line.find(":") + 2));

  getline(in, line);

  node.resize(size);

  for (size_t i = 0; i < size; ++i) {
    getline(in, line);

    stringstream iss(line);
    size_t id;
    string xStr, yStr;

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
  in.close();

  return capacity;
}

void VRP::print() {
  std::cout << "DIMENSION:" << size << '\n';
  std::cout << "CAPACITY:" << capacity << '\n';
  for (auto i = 0u; i < size; ++i) {
    std::cout << i << ':'
              << setw(6) << node[i].x << ' '
              << setw(6) << node[i].y << ' '
              << setw(6) << node[i].demand << std::endl;
  }
}

std::vector<std::vector<Edge>>
PrimsAlgo(const VRP &vrp, std::vector<std::vector<Edge>> &graph) {
  auto N = graph.size();
  const node_t INIT = -1;

  std::vector<weight_t> key(N, INT_MAX);
  std::vector<weight_t> toEdges(N, -1);
  std::vector<bool> visited(N, false);

  std::set<std::pair<weight_t, node_t>> active;
  std::vector<std::vector<Edge>> nG(N);

  node_t src = 0;
  key[src] = 0.0;
  active.insert({0.0, src});

  while (active.size() > 0) {
    auto where = active.begin()->second;

    active.erase(active.begin());
    if (visited[where]) {
      continue;
    }
    visited[where] = true;
    for (Edge E : graph[where]) {
      if (!visited[E.to] && E.length < key[E.to]) {
        key[E.to] = E.length;
        active.insert({key[E.to], E.to});

        toEdges[E.to] = where;
      }
    }
  }

  node_t u = 0;
  for (auto v : toEdges) {
    if (v != INIT) {

      weight_t w = vrp.get_dist(u, v);

      nG[u].push_back(Edge(v, w));
      nG[v].push_back(Edge(u, w));

      DEBUG std::cout << u << " -- " << v << '\n';
    }
    u++;
  }
  return nG;
}

void printAdjList(const std::vector<std::vector<Edge>> &graph) {
  int i = 0;
  for (auto vec : graph) {
    std::cout << i << ": ";
    for (auto e : vec) {
      std::cout << e.to << " ";
    }
    i++;
    std::cout << std::endl;
  }
}

void ShortCircutTour(std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t u, std::vector<node_t> &out) {
  visited[u] = true;
  DEBUG std::cout << u << ' ';

  out.push_back(u);
  for (auto e : g[u]) {
    node_t v = e.to;
    if (!visited[v]) {
      ShortCircutTour(g, visited, v, out);
    }
  }
}

std::vector<std::vector<node_t>>
convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute) {
  std::vector<std::vector<node_t>> routes;

  demand_t vCapacity = vrp.getCapacity();
  demand_t residueCap = vCapacity;
  std::vector<node_t> aRoute;

  for (auto v : singleRoute) {
    if (v == 0)
      continue;
    if (residueCap - vrp.node[v].demand >= 0) {
      aRoute.push_back(v);
      residueCap = residueCap - vrp.node[v].demand;
    } else {
      routes.push_back(aRoute);
      aRoute.clear();
      aRoute.push_back(v);
      residueCap = vCapacity - vrp.node[v].demand;
    }
  }
  routes.push_back(aRoute);
  return routes;
}

weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute, node_t depot = 1) {
  weight_t routeVal = 0;
  node_t prevPoint = 0;

  for (auto aPoint : aRoute) {
    routeVal += vrp.get_dist(prevPoint, aPoint);
    prevPoint = aPoint;
  }
  routeVal += vrp.get_dist(prevPoint, 0);

  return routeVal;
}

void printOutput(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;

  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    std::cout << "Route #" << ii + 1 << ":";
    for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj) {
      std::cout << " " << final_routes[ii][jj];
    }
    std::cout << '\n';
  }

  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);

    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    }
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

    total_cost += curr_route_cost;
  }

  std::cout << "Cost " << total_cost << std::endl;
}

void tsp_approx(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, node_t ncities) {
  node_t i, j;
  node_t ClosePt = 0;
  weight_t CloseDist;

  for (i = 1; i < ncities; i++)
    tour[i] = cities[i - 1];

  tour[0] = cities[ncities - 1];

  for (i = 1; i < ncities; i++) {
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    CloseDist = DBL_MAX;
    for (j = ncities - 1;; j--) {
      weight_t ThisDist = (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
      if (ThisDist <= CloseDist) {
        ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
        if (ThisDist <= CloseDist) {
          if (j < i)
            break;
          CloseDist = ThisDist;
          ClosePt = j;
        }
      }
    }

    unsigned temp = tour[i];
    tour[i] = tour[ClosePt];
    tour[ClosePt] = temp;
  }
}

std::vector<std::vector<node_t>>
postprocess_tsp_approx(const VRP &vrp, std::vector<std::vector<node_t>> &solRoutes) {
  std::vector<std::vector<node_t>> modifiedRoutes;

  unsigned nroutes = solRoutes.size();
  for (unsigned i = 0; i < nroutes; ++i) {

    unsigned sz = solRoutes[i].size();
    std::vector<node_t> cities(sz + 1);
    std::vector<node_t> tour(sz + 1);

    for (unsigned j = 0; j < sz; ++j)
      cities[j] = solRoutes[i][j];

    cities[sz] = 0;

    tsp_approx(vrp, cities, tour, sz + 1);

    vector<node_t> curr_route;
    for (unsigned kk = 1; kk < sz + 1; ++kk) {
      curr_route.push_back(tour[kk]);
    }

    modifiedRoutes.push_back(curr_route);
  }
  return modifiedRoutes;
}

void tsp_2opt(const VRP &vrp, std::vector<node_t> &cities, std::vector<node_t> &tour, unsigned ncities) {

  unsigned improve = 0;

  while (improve < 2) {
    double best_distance = 0.0;

    best_distance += vrp.get_dist(DEPOT, cities[0]);

    for (unsigned jj = 1; jj < ncities; ++jj) {

      best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
    }

    best_distance += vrp.get_dist(DEPOT, cities[ncities - 1]);

    for (unsigned i = 0; i < ncities - 1; i++) {
      for (unsigned k = i + 1; k < ncities; k++) {
        for (unsigned c = 0; c < i; ++c) {
          tour[c] = cities[c];
        }

        unsigned dec = 0;
        for (unsigned c = i; c < k + 1; ++c) {
          tour[c] = cities[k - dec];
          dec++;
        }

        for (unsigned c = k + 1; c < ncities; ++c) {
          tour[c] = cities[c];
        }
        double new_distance = 0.0;

        new_distance += vrp.get_dist(DEPOT, tour[0]);
        for (unsigned jj = 1; jj < ncities; ++jj) {

          new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
        }

        new_distance += vrp.get_dist(DEPOT, tour[ncities - 1]);

        if (new_distance < best_distance) {

          improve = 0;
          for (unsigned jj = 0; jj < ncities; jj++)
            cities[jj] = tour[jj];
          best_distance = new_distance;
        }
      }
    }
    improve++;
  }
}

std::vector<std::vector<node_t>>
postprocess_2OPT(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes) {
  std::vector<std::vector<node_t>> postprocessed_final_routes;

  unsigned nroutes = final_routes.size();
  for (unsigned i = 0; i < nroutes; ++i) {

    unsigned sz = final_routes[i].size();

    std::vector<node_t> cities(sz);
    std::vector<node_t> tour(sz);

    for (unsigned j = 0; j < sz; ++j)
      cities[j] = final_routes[i][j];

    vector<node_t> curr_route;

    if (sz > 2)
      tsp_2opt(vrp, cities, tour, sz);

    for (unsigned kk = 0; kk < sz; ++kk) {
      curr_route.push_back(cities[kk]);
    }

    postprocessed_final_routes.push_back(curr_route);
  }
  return postprocessed_final_routes;
}

weight_t get_total_cost_of_routes(const VRP &vrp, vector<vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {

      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    }

    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);

    total_cost += curr_route_cost;
  }

  return total_cost;
}

std::vector<std::vector<node_t>>
postProcessIt(const VRP &vrp, std::vector<std::vector<node_t>> &final_routes, weight_t &minCost) {
  std::vector<std::vector<node_t>> postprocessed_final_routes(final_routes.size());

  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);

#pragma omp parallel for
  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz) {

    vector<node_t> postprocessed_route2 = postprocessed_final_routes2[zzz];
    vector<node_t> postprocessed_route3 = postprocessed_final_routes3[zzz];

    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();

    weight_t postprocessed_route2_cost = 0.0;

    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
    for (unsigned jj = 1; jj < sz2; ++jj) {

      postprocessed_route2_cost += vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    }

    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

    weight_t postprocessed_route3_cost = 0.0;

    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj) {

      postprocessed_route3_cost += vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    }

    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

    if (postprocessed_route3_cost > postprocessed_route2_cost) {
      postprocessed_final_routes[zzz] = postprocessed_route2;
    }

    else {
      postprocessed_final_routes[zzz] = postprocessed_route3;
    }
  }

  auto postprocessed_final_routes_cost = get_total_cost_of_routes(vrp, postprocessed_final_routes);

  minCost = postprocessed_final_routes_cost;

  return postprocessed_final_routes;
}

std::pair<weight_t, std::vector<std::vector<node_t>>>
calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;

#pragma omp parallel for reduction(+ : total_cost)
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    weight_t curr_route_cost = 0;
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][0]);

#pragma omp parallel for reduction(+ : curr_route_cost)
    for (unsigned jj = 1; jj < final_routes[ii].size(); ++jj) {
      curr_route_cost += vrp.get_dist(final_routes[ii][jj - 1], final_routes[ii][jj]);
    }
    curr_route_cost += vrp.get_dist(DEPOT, final_routes[ii][final_routes[ii].size() - 1]);
    total_cost += curr_route_cost;
  }
  return {total_cost, final_routes};
}

bool verify_sol(const VRP &vrp, vector<vector<node_t>> final_routes, unsigned capacity) {

  unsigned *hist = (unsigned *)malloc(sizeof(unsigned) * vrp.getSize());
  memset(hist, 0, sizeof(unsigned) * vrp.getSize());

  for (unsigned i = 0; i < final_routes.size(); ++i) {
    unsigned route_sum_of_demands = 0;
    for (unsigned j = 0; j < final_routes[i].size(); ++j) {

      route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
      hist[final_routes[i][j]] += 1;
    }
    if (route_sum_of_demands > capacity) {
      return false;
    }
  }

  for (unsigned i = 1; i < vrp.getSize(); ++i) {
    if (hist[i] > 1) {
      return false;
    }
    if (hist[i] == 0) {
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 2) {
    std::cout << "parMDS version 1.1" << '\n';
    std::cout << "Usage: " << argv[0] << " toy.vrp [-nthreads <n> DEFAULT is 20] [-round 0 or 1 DEFAULT:1]" << '\n';
    exit(1);
  }

  for (int ii = 2; ii < argc; ii += 2) {
    if (std::string(argv[ii]) == "-round")
      vrp.params.toRound = atoi(argv[ii + 1]);
    else if (std::string(argv[ii]) == "-nthreads")
      vrp.params.nThreads = atoi(argv[ii + 1]);
    else {
      std::cerr << "INVALID Arguments!" << '\n';
      std::cerr << "Usage:" << argv[0] << " toy.vrp -nthreads 20 -round 1" << '\n';
      exit(1);
    }
  }

  vrp.read(argv[1]);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

  auto cG = vrp.cal_graph_dist();

  auto mstG = PrimsAlgo(vrp, cG);

  std::vector<bool> visited(mstG.size(), false);
  visited[0] = true;
  std::vector<int> singleRoute;

  weight_t minCost = INT_MAX * 1.0f;
  std::vector<std::vector<node_t>> minRoute;

  const auto baseMst = mstG;

  for (int i = 0; i < 1; i++) {
    auto mstCopy = baseMst;

    for (auto &list : mstCopy) {
      std::shuffle(list.begin(), list.end(), std::default_random_engine(0));
    }

    std::vector<int> singleRoute;

    std::vector<bool> visited(mstCopy.size(), false);
    visited[0] = true;

    ShortCircutTour(mstCopy, visited, 0, singleRoute);
    DEBUG std::cout << '\n';

    auto aRoutes = convertToVrpRoutes(vrp, singleRoute);

    auto aCostRoute = calCost(vrp, aRoutes);
    if (aCostRoute.first < minCost) {
      minCost = aCostRoute.first;
      minRoute = aCostRoute.second;
    }
  }

  auto minCost1 = minCost;

  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  auto timeUpto1 = (double)(elapsed * 1.E-9);

  short PARLIMIT = vrp.params.nThreads;

#pragma omp parallel num_threads(PARLIMIT)
  {
    weight_t threadMinCost = minCost;
    std::vector<std::vector<node_t>> threadMinRoute = minRoute;

#pragma omp for
  for (int i = 0; i < 100000; i += PARLIMIT) {
    auto mstCopy = baseMst;
    std::seed_seq seed{12345, i};
    std::mt19937 rng(seed);

    for (auto &list : mstCopy) {
      std::shuffle(list.begin(), list.end(), rng);
    }

    std::vector<int> singleRoute;

    std::vector<bool> visited(mstCopy.size(), false);
    visited[0] = true;

    ShortCircutTour(mstCopy, visited, 0, singleRoute);
    DEBUG std::cout << '\n';

    auto aRoutes = convertToVrpRoutes(vrp, singleRoute);

    auto aCostRoute = calCost(vrp, aRoutes);
    if (aCostRoute.first < threadMinCost) {
      threadMinCost = aCostRoute.first;
      threadMinRoute = aCostRoute.second;
    }
  }

#pragma omp critical
    {
      if (threadMinCost < minCost) {
        minCost = threadMinCost;
        minRoute = threadMinRoute;
      }
    }
  }

  auto minCost2 = minCost;
  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  auto timeUpto2 = (double)(elapsed * 1.E-9);

  auto postRoutes = postProcessIt(vrp, minRoute, minCost);

  end = std::chrono::high_resolution_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  double total_time = (double)(elapsed * 1.E-9);

  bool verified = false;
  verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

  std::cerr << argv[1] << " Cost ";
  std::cerr << minCost1 << ' ';
  std::cerr << minCost2 << ' ';
  std::cerr << minCost;

  std::cerr << " Time(seconds) ";
  std::cerr << timeUpto1 << ' ';
  std::cerr << timeUpto2 << ' ';
  std::cerr << total_time;

  std::cerr << " parLimit " << PARLIMIT;

  if (verified)
    std::cerr << " VALID" << std::endl;
  else
    std::cerr << " INVALID" << std::endl;

  printOutput(vrp, postRoutes);

  return 0;
}
