CUDA_ARCH = -arch=sm_86
CXXFLAGS  = -O3 -std=c++14 -Xcompiler -fopenmp
NVCC      = nvcc

TARGET = parMDS.out

SRCS = parMDS.cpp gpuWrapper.cu gpuEval.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) $(SRCS) -o $(TARGET)

run:
	./$(TARGET) inputs/Antwerp1.vrp -nthreads 16 -round 1

clean:
	rm -f *.out

