CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
OMPFLAGS = -fopenmp
NVCC = nvcc
CUDAFLAGS = -O3 -arch=sm_60

# Define source and include directories
SRCDIR = src
INCDIR = include
# AI EDIT: Add CUDA paths for hybrid compilation
CUDA_PATH = /usr/local/cuda
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64
# END AI EDIT

# Add the include path to CXXFLAGS and CUDAFLAGS
# The -I flag tells the compiler to look in $(INCDIR) for header files
CXXFLAGS += -I $(INCDIR)
CUDAFLAGS += -I $(INCDIR)


# Week 1 targets
serial: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -o ray_serial $(SRCDIR)/main.cpp

openmp: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o ray_openmp $(SRCDIR)/main.cpp

# Week 2 target
cuda: $(SRCDIR)/main_gpu.cu
	$(NVCC) $(CUDAFLAGS) -o ray_cuda $(SRCDIR)/main_gpu.cu

# Week 3 target (placeholder)
# AI EDIT: Add CUDA include path and library linking for hybrid
hybrid: $(SRCDIR)/main_hybrid.cpp $(SRCDIR)/kernel.cu
	$(NVCC) $(CUDAFLAGS) -c $(SRCDIR)/kernel.cu
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(CUDA_INC) -c $(SRCDIR)/main_hybrid.cpp
	$(NVCC) $(CUDAFLAGS) kernel.o main_hybrid.o -o ray_hybrid -L$(CUDA_LIB) -lcudart
# END AI EDIT

clean:
	rm -f ray_serial ray_openmp ray_cuda ray_hybrid *.o *.ppm

test: serial
	./ray_serial
	@echo "Check output_serial.ppm"

benchmark: serial openmp
	@echo "=== Performance Comparison ==="
	@echo -n "Serial: "; ./ray_serial | grep "Serial time"
	@echo -n "OpenMP: "; ./ray_openmp | grep "OpenMP time"

benchmark2: serial openmp
	@echo "=== Performance Comparison ==="
	@echo ""
	@echo "--- Simple Scene ---"
	@SERIAL_TIME=$$(./ray_serial scenes/simple.txt 2>&1 | grep -oP 'Serial time: \K[0-9.]+'); \
	OPENMP_TIME=$$(./ray_openmp scenes/simple.txt 2>&1 | grep -oP 'OpenMP time: \K[0-9.]+'); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk "BEGIN {printf \"%.2f\", $$SERIAL_TIME / $$OPENMP_TIME}"); \
	echo "Speedup: $${SPEEDUP}x"
	@mv output_serial.ppm output_simple_serial.ppm
	@mv output_openmp.ppm output_simple_openmp.ppm
# 	@convert output_simple_serial.ppm output_simple_serial.png
# 	@convert output_simple_openmp.ppm output_simple_openmp.png
	@echo ""
	@echo "--- Medium Scene ---"
	@SERIAL_TIME=$$(./ray_serial scenes/medium.txt 2>&1 | grep -oP 'Serial time: \K[0-9.]+'); \
	OPENMP_TIME=$$(./ray_openmp scenes/medium.txt 2>&1 | grep -oP 'OpenMP time: \K[0-9.]+'); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk "BEGIN {printf \"%.2f\", $$SERIAL_TIME / $$OPENMP_TIME}"); \
	echo "Speedup: $${SPEEDUP}x"
	@mv output_serial.ppm output_medium_serial.ppm
	@mv output_openmp.ppm output_medium_openmp.ppm
# 	@convert output_medium_serial.ppm output_medium_serial.png
# 	@convert output_medium_openmp.ppm output_medium_openmp.png
	@echo ""
	@echo "--- Complex Scene ---"
	@SERIAL_TIME=$$(./ray_serial scenes/complex.txt 2>&1 | grep -oP 'Serial time: \K[0-9.]+'); \
	OPENMP_TIME=$$(./ray_openmp scenes/complex.txt 2>&1 | grep -oP 'OpenMP time: \K[0-9.]+'); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk "BEGIN {printf \"%.2f\", $$SERIAL_TIME / $$OPENMP_TIME}"); \
	echo "Speedup: $${SPEEDUP}x"
	@mv output_serial.ppm output_complex_serial.ppm
	@mv output_openmp.ppm output_complex_openmp.ppm
# 	@convert output_complex_serial.ppm output_complex_serial.png
# 	@convert output_complex_openmp.ppm output_complex_openmp.png
	@echo ""
	@echo "=== Benchmark Complete ==="