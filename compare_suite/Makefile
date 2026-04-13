NVCC      := nvcc
CXX       := g++
CXXFLAGS  := -O3 -std=c++14
OMPFLAGS  := -fopenmp
CUDA_OMP  := -Xcompiler $(OMPFLAGS)

ALL_BINS := seqMDS.out parMDS.out gpuMDS.out gpucpuMDS.out gpucpuMDS_v2.out

.PHONY: all clean

all: $(ALL_BINS)

seqMDS.out: seqMDS.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

parMDS.out: parMDS.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

gpuMDS.out: gpuMDS.cu
	$(NVCC) -O3 -std=c++14 $< -o $@

gpucpuMDS.out: gpucpuMDS.cu
	$(NVCC) -O3 -std=c++14 $(CUDA_OMP) $< -o $@

gpucpuMDS_v2.out: gpucpuMDS_v2.cu
	$(NVCC) -O3 -std=c++14 $(CUDA_OMP) $< -o $@

clean:
	rm -f $(ALL_BINS)
