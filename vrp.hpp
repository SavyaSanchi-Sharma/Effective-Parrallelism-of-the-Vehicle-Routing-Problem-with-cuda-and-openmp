#ifndef VRP_HPP
#define VRP_HPP

#include <vector>
#include <string>
#include <cstddef>

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;

class Params {
public:
    bool toRound;
    short nThreads;
    Params() : toRound(1), nThreads(20) {}
};

class Edge {
public:
    node_t to;
    weight_t length;
    Edge() {}
    Edge(node_t t, weight_t l) : to(t), length(l) {}
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
    std::string type;

public:
    std::vector<Point> node;
    std::vector<weight_t> dist;
    Params params;

    size_t getSize() const { return size; }
    demand_t getCapacity() const { return capacity; }

    unsigned read(std::string filename);
    void print();
    void print_dist();
    std::vector<std::vector<Edge>> cal_graph_dist();
    weight_t get_dist(node_t i, node_t j) const;
};

#endif
