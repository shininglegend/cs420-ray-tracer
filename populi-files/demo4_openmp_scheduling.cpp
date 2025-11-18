// demo4_openmp_scheduling.cpp
// Compile: g++ -fopenmp -o demo4 demo4_openmp_scheduling.cpp
// Usage: ./demo4

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cstring>

// Simulate ray tracing work with variable cost
void simulate_pixel_work(int x, int y, int complexity) {
    // Simulate variable work based on position
    // Edges are cheap, center is expensive (more sphere intersections)
    int center_x = 320, center_y = 240;
    int dist = abs(x - center_x) + abs(y - center_y);
    
    // Work amount varies by distance from center
    int work_amount = complexity * (500 - dist);
    
    // Simulate computation
    volatile double sum = 0;
    for (int i = 0; i < work_amount; i++) {
        sum += sin(i) * cos(i);
    }
}

void render_with_schedule(const char* schedule_name, int width, int height) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> thread_times(omp_get_max_threads(), 0.0);
    
    if (strcmp(schedule_name, "static") == 0) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto thread_start = std::chrono::high_resolution_clock::now();
                simulate_pixel_work(x, y, 1);
                auto thread_end = std::chrono::high_resolution_clock::now();
                
                int tid = omp_get_thread_num();
                std::chrono::duration<double> diff = thread_end - thread_start;
                thread_times[tid] += diff.count();
            }
        }
    } else if (strcmp(schedule_name, "dynamic") == 0) {
        #pragma omp parallel for schedule(dynamic, 16) collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto thread_start = std::chrono::high_resolution_clock::now();
                simulate_pixel_work(x, y, 1);
                auto thread_end = std::chrono::high_resolution_clock::now();
                
                int tid = omp_get_thread_num();
                std::chrono::duration<double> diff = thread_end - thread_start;
                thread_times[tid] += diff.count();
            }
        }
    } else if (strcmp(schedule_name, "guided") == 0) {
        #pragma omp parallel for schedule(guided) collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto thread_start = std::chrono::high_resolution_clock::now();
                simulate_pixel_work(x, y, 1);
                auto thread_end = std::chrono::high_resolution_clock::now();
                
                int tid = omp_get_thread_num();
                std::chrono::duration<double> diff = thread_end - thread_start;
                thread_times[tid] += diff.count();
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_diff = end - start;
    
    std::cout << "\n" << schedule_name << " scheduling:\n";
    std::cout << "  Total time: " << total_diff.count() << " seconds\n";
    
    // Calculate load imbalance
    double max_thread_time = 0, min_thread_time = 1e9;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        if (thread_times[i] > max_thread_time) max_thread_time = thread_times[i];
        if (thread_times[i] < min_thread_time && thread_times[i] > 0) 
            min_thread_time = thread_times[i];
    }
    
    double imbalance = (max_thread_time - min_thread_time) / max_thread_time * 100;
    std::cout << "  Load imbalance: " << imbalance << "%\n";
}

int main() {
    std::cout << "=== OpenMP Scheduling Comparison ===\n";
    std::cout << "Simulating 640x480 image with variable work per pixel\n";
    std::cout << "(Center pixels are more expensive - more spheres to test)\n\n";
    
    const int width = 640;
    const int height = 480;
    
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads\n";
    
    // Test different scheduling strategies
    render_with_schedule("static", width, height);
    render_with_schedule("dynamic", width, height);
    render_with_schedule("guided", width, height);
    
    std::cout << "\nConclusion:\n";
    std::cout << "- Static: Fast but poor load balance for variable work\n";
    std::cout << "- Dynamic: Better balance but scheduling overhead\n";
    std::cout << "- Guided: Good compromise for ray tracing workload\n";
    
    return 0;
}