Findings

  1. mstCopy is shared across all OpenMP iterations and mutated concurrently.
     In the parallel superloop, every thread shuffles the same adjacency lists in mstCopy:

  upstream-parMDS/parMDS.cpp:762
  upstream-parMDS/parMDS.cpp:764

  That is a data race. Multiple threads are reordering the same vectors at the same time, so DFS can read a corrupted or partially
  shuffled MST.

  2. minCost and minRoute are updated without synchronization.
     Every thread compares and then writes to shared best-solution state:

  upstream-parMDS/parMDS.cpp:762
  upstream-parMDS/parMDS.cpp:781
  upstream-parMDS/parMDS.cpp:782
  upstream-parMDS/parMDS.cpp:784

  This can lose the true minimum, pair the wrong route with a cost, or leave minRoute partially overwritten.

  3. postProcessIt() pushes into one shared vector from a parallel loop.
     This is another hard race:

  upstream-parMDS/parMDS.cpp:576
  upstream-parMDS/parMDS.cpp:608
  upstream-parMDS/parMDS.cpp:612

  postprocessed_final_routes.push_back(...) from multiple threads is undefined behavior. This alone can explain missing routes like
  Route #1: 3.

  4. rand() is used inside parallel code.
     rand() is not suitable shared RNG state for threaded randomization:

  upstream-parMDS/parMDS.cpp:765

  Even if the container races were fixed, this random source is still weak and potentially racy depending on libc/runtime behavior.

  What is probably happening in your run

  The most direct cause of INVALID and a single-route output is:

  - superloop corruption from shared mstCopy and shared best-solution state
  - then postProcessIt() corrupts the final route vector further with parallel push_back

  So this is not a compiler-flag issue. It is a code-level concurrency issue.

  Minimum fixes needed to make it sane

  1. Make MST local per iteration:

  - replace shared mstCopy access with a thread-local copy inside the loop

  2. Make best-solution update safe:

  - keep thread-local best (cost, routes)
  - merge into global best in a critical section or after the loop

  3. Fix postProcessIt():

  - pre-size the output vector and assign by index instead of push_back in parallel

  4. Replace rand():

  - use a thread-local std::mt19937

  Bottom line

  The original parMDS.cpp is not safe as written under OpenMP with g++ -fopenmp. The core issues are unsynchronized shared mutation
  of:

  - mstCopy
  - minCost
  - minRoute
  - postprocessed_final_routes
