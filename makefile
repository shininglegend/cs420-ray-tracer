CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
OMPFLAGS = -fopenmp
NVCC = nvcc
CUDAFLAGS = -O3 -arch=sm_60

# Week 1 targets
serial: main.cpp
	$(CXX) $(CXXFLAGS) -o ray_serial main.cpp

openmp: main.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o ray_openmp main.cpp

# Week 2 target (placeholder)
cuda: main_gpu.cu
	$(NVCC) $(CUDAFLAGS) -o ray_cuda main_gpu.cu

# Week 3 target (placeholder)
hybrid: main_hybrid.cpp kernel.cu
	$(NVCC) $(CUDAFLAGS) -c kernel.cu
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c main_hybrid.cpp
	$(NVCC) $(CUDAFLAGS) kernel.o main_hybrid.o -o ray_hybrid

clean:
	rm -f ray_serial ray_openmp ray_cuda ray_hybrid *.o *.ppm

test: serial
	./ray_serial
	@echo "Check output_serial.ppm"

benchmark: serial openmp
	@echo "=== Performance Comparison ==="
	@echo -n "Serial: "; ./ray_serial | grep "Serial time"
	@echo -n "OpenMP: "; ./ray_openmp | grep "OpenMP time"