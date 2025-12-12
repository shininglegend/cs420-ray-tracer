// main_hybrid.cpp - Week 3: Hybrid CPU-GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "camera.h"
#include "ray.h"
#include "ray_math_constants.h"
#include "scene.h"
#include "scene_loader.h"
#include "sphere.h"
#include "vec3.h"

// CUDA runtime API (for hybrid execution)
#include "gpu_shared.h"
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =========================================================
// Image Output Functions
// =========================================================

void write_ppm(const std::string &filename,
               const std::vector<Vec3> &framebuffer, int width, int height) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return;
  }

  // PPM header
  file << "P3\n" << width << " " << height << "\n255\n";

  // Write pixels (PPM is top-to-bottom)
  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      Vec3 color = framebuffer[j * width + i];
      auto r = color.x;
      auto g = color.y;
      auto b = color.z;
      r = int(255.99 * std::min(1.0, r));
      g = int(255.99 * std::min(1.0, g));
      b = int(255.99 * std::min(1.0, b));
      // framebuffer[j * width + i].to_rgb(r, g, b);
      file << r << " " << g << " " << b << "\n";
    }
  }

  file.close();
  std::cout << "Image written to " << filename << std::endl;
}

// =========================================================
// Scene Creation (reuse from Week 1)
// =========================================================
// NOTE FROM TITUS: This loads from scene_loader.h instead.

// =========================================================
// Tile Structure for Work Distribution
// =========================================================
struct Tile {
  int x_start, y_start;
  int x_end, y_end;
  int complexity_estimate; // Estimated work for this tile
  bool processed;

  Tile(int xs, int ys, int xe, int ye)
      : x_start(xs), y_start(ys), x_end(xe), y_end(ye), complexity_estimate(0),
        processed(false) {}

  int pixel_count() const { return (x_end - x_start) * (y_end - y_start); }
};

// =========================================================
// GPU Kernel Declaration (implemented in kernel.cu)
// =========================================================
// BEGIN AI EDIT: Fix declaration to match call site and use float3*
extern "C" void launch_gpu_kernel(float3 *d_framebuffer, GPUSphere *d_spheres,
                                  int num_spheres, int num_lights, int tile_x,
                                  int tile_y, int tile_width, int tile_height,
                                  int image_width, int image_height,
                                  int max_depth, cudaStream_t stream);

// =========================================================
// CPU Ray Tracing (Complex Shading Path)
// =========================================================
Vec3 trace_ray_cpu(const Ray &ray, const Scene &scene, int depth) {
  // TODO: STUDENT - Implement CPU ray tracing
  // This should handle complex shading, deep reflections, etc.
  // Can reuse code from Week 1
  // NOTE (TM): This is a pretty much direct copy-paste from week 1, but I
  // removed some comments to shorten it up.
  if (depth <= 0)
    return Vec3(0, 0, 0);
  double t;
  int sphere_idx;

  // 1. Calculate hit point (scene.h.intersection)
  if (!scene.find_intersection(ray, t, sphere_idx)) {
    // Sky color gradient
    double t = 0.5 * (ray.direction.y + 1.0);
    return Vec3(1, 1, 1) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
  }

  Vec3 hit = ray.origin + ray.direction * t;            // Find actual hitpooint
  Vec3 norm = scene.spheres[sphere_idx].normal_at(hit); // Normalize

  // 2. Call scene.shade() for color
  Vec3 view_dir = (ray.origin - hit).normalized(); // From Ai
  Vec3 shade =
      scene.shade(hit, norm, scene.spheres[sphere_idx].material, view_dir);

  // 3. If material is reflective, recursively trace reflection ray
  if (scene.spheres[sphere_idx].material.reflectivity > 0) {
    Vec3 reflected_dir = ray.direction - norm * 2.0 * dot(ray.direction, norm);
    Ray reflected_ray(hit + norm * EPSILON, reflected_dir);
    Vec3 reflected_color = trace_ray_cpu(reflected_ray, scene, depth - 1);

    double refl = scene.spheres[sphere_idx].material.reflectivity;
    shade = shade * (1.0 - refl) + reflected_color * refl;
  }
  return shade;
}

void process_tile_cpu(const Tile &tile, const Scene &scene,
                      const Camera &camera, std::vector<Vec3> &framebuffer,
                      int width, int height, int max_depth) {

#pragma omp parallel for collapse(2) schedule(dynamic, 4)
  for (int y = tile.y_start; y < tile.y_end; y++) {
    for (int x = tile.x_start; x < tile.x_end; x++) {
      double u = double(x) / (width - 1);
      double v = double(y) / (height - 1);
      Ray ray = camera.get_ray(u, v);
      framebuffer[y * width + x] = trace_ray_cpu(ray, scene, max_depth);
    }
  }
}

// =========================================================
// GPU Memory Management
// =========================================================
extern "C" void upload_lights(GPULight *lights, int count);

class GPUResources {
private:
  float3 *d_framebuffer;
  GPUSphere *d_spheres;
  size_t fb_size;
  size_t spheres_size;

public:
  GPUResources(int width, int height, int num_spheres, int num_lights) {
    fb_size = width * height * sizeof(float3);
    spheres_size = num_spheres * sizeof(GPUSphere); // Switch to object

    CUDA_CHECK(cudaMalloc(&d_framebuffer, fb_size));
    CUDA_CHECK(cudaMalloc(&d_spheres, spheres_size));
  }

  ~GPUResources() {
    CUDA_CHECK(cudaFree(d_framebuffer));
    CUDA_CHECK(cudaFree(d_spheres));
  }

  void upload_scene(const Scene &scene) {
    // TODO: STUDENT - Convert scene data to GPU format and upload
    // Pack spheres and lights into float arrays
    std::vector<GPUSphere> h_spheres;
    std::vector<GPULight> h_lights;

    // Convert spheres
    for (const auto &sphere : scene.spheres) {
      GPUSphere gpu_sphere;
      gpu_sphere.center =
          make_float3(sphere.center.x, sphere.center.y, sphere.center.z);
      gpu_sphere.radius = sphere.radius;
      gpu_sphere.material.albedo =
          make_float3(sphere.material.color.x, sphere.material.color.y,
                      sphere.material.color.z);
      gpu_sphere.material.metallic = sphere.material.reflectivity;
      gpu_sphere.material.shininess = sphere.material.shininess;
      h_spheres.push_back(gpu_sphere);
    }
    // This line was helpfully provided by Ai
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(),
                          h_spheres.size() * sizeof(GPUSphere),
                          cudaMemcpyHostToDevice));

    // Convert lights
    for (const auto &light : scene.lights) {
      GPULight gpu_light;
      gpu_light.position =
          make_float3(light.position.x, light.position.y, light.position.z);
      gpu_light.color =
          make_float3(light.color.x, light.color.y, light.color.z);
      gpu_light.intensity = light.intensity;
      h_lights.push_back(gpu_light);
    }
    // CUDA_CHECK(cudaMemcpyToSymbol(const_lights, h_lights.data(),
    //                               h_lights.size() * sizeof(GPULight)));
    upload_lights(h_lights.data(), h_lights.size());
  }

  void download_tile(const Tile &tile, std::vector<Vec3> &framebuffer,
                     int width, cudaStream_t stream) {
    // set up a temp framebuffer
    size_t tile_height = tile.y_end - tile.y_start;
    size_t tile_width = tile.x_end - tile.x_start;
    std::vector<float3> temp_framebuffer(tile_height * tile_width);
    // Ai helped with pseudocode and debugging for this
    // copy each row
    for (int y = tile.y_start; y < tile.y_end; y++) {
      size_t row_offset = y * width + tile.x_start;
      size_t temp_offset = (y - tile.y_start) * tile_width;
      // copy it down
      CUDA_CHECK(cudaMemcpyAsync(
          &temp_framebuffer[temp_offset], &d_framebuffer[row_offset],
          tile_width * sizeof(float3), cudaMemcpyDeviceToHost, stream));
    }
    // Sync
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Then convert all rows to float3 → Vec3
    for (int y = tile.y_start; y < tile.y_end; y++) {
      for (int x = tile.x_start; x < tile.x_end; x++) {
        int temp_idx = (y - tile.y_start) * tile_width + (x - tile.x_start);
        int fb_idx = y * width + x;
        float3 gpu_color = temp_framebuffer[temp_idx];
        framebuffer[fb_idx] = Vec3(gpu_color.x, gpu_color.y, gpu_color.z);
      }
    }
  }

  float3 *get_framebuffer() { return d_framebuffer; }
  GPUSphere *get_spheres() { return d_spheres; }
};

// =========================================================
// Tile Complexity Estimation
// =========================================================
int estimate_tile_complexity(const Tile &tile, const Scene &scene,
                             const Camera &camera) {
  // TODO: STUDENT - Implement heuristic to estimate rendering complexity
  // Consider:
  // - Number of spheres likely to be intersected
  // - Presence of reflective materials
  // - Distance from camera
  // Simple version: sample a few rays and count intersections

  return tile.pixel_count(); // Placeholder: just use pixel count
}

// =========================================================
// TODO: STUDENT IMPLEMENTATION - Hybrid Work Distribution
// =========================================================
// Design and implement the work distribution strategy.
// Decide which tiles go to CPU vs GPU based on:
// - Complexity estimates
// - Current workload
// - Memory constraints
// Use CUDA streams for overlapping computation
// =========================================================

void render_hybrid(const Scene &scene, const Camera &camera,
                   std::vector<Vec3> &framebuffer, int width, int height,
                   int max_depth, int tile_size = 64) {

  std::cout << "Hybrid Rendering..." << std::endl;

  // Create tiles
  std::vector<Tile> tiles;
  for (int y = 0; y < height; y += tile_size) {
    for (int x = 0; x < width; x += tile_size) {
      int xe = std::min(x + tile_size, width);
      int ye = std::min(y + tile_size, height);
      tiles.emplace_back(x, y, xe, ye);
    }
  }

  std::cout << "Created " << tiles.size() << " tiles of size " << tile_size
            << "x" << tile_size << std::endl;

  // Estimate complexity for each tile
  for (auto &tile : tiles) {
    tile.complexity_estimate = estimate_tile_complexity(tile, scene, camera);
  }

  // Initialize GPU resources
  GPUResources gpu_resources(width, height, scene.spheres.size(),
                             scene.lights.size());
  gpu_resources.upload_scene(scene);

  // Create CUDA streams for pipelining
  const int NUM_STREAMS = 3;
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // TODO: STUDENT - Implement work distribution
  // Split tiles between CPU and GPU based on complexity
  std::queue<Tile *> cpu_queue;
  std::queue<Tile *> gpu_queue;

  // Simple strategy: complex tiles to CPU, simple to GPU
  int complexity_threshold = width * height / (tile_size * tile_size) * 2;

  for (auto &tile : tiles) {
    // if (tile.complexity_estimate > complexity_threshold) {
    if (false) {
      cpu_queue.push(&tile);
    } else {
      gpu_queue.push(&tile);
    }
  }

  std::cout << "Distribution: " << cpu_queue.size() << " tiles to CPU, "
            << gpu_queue.size() << " tiles to GPU" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

// TODO: STUDENT - Process tiles concurrently
// Use OpenMP sections or std::thread for CPU-GPU concurrency
#pragma omp parallel sections
  {
// CPU processing section
#pragma omp section
    {
      while (!cpu_queue.empty()) {
        Tile *tile = cpu_queue.front();
        cpu_queue.pop();
        process_tile_cpu(*tile, scene, camera, framebuffer, width, height,
                         max_depth);
        tile->processed = true;
      }
    }

// GPU processing section
#pragma omp section
    {
      int stream_idx = 0;
      while (!gpu_queue.empty()) {
        Tile *tile = gpu_queue.front();
        gpu_queue.pop();

        // TODO: STUDENT - Launch GPU kernel for this tile
        // Use streams for asynchronous execution
        cudaStream_t stream = streams[stream_idx];
        stream_idx = (stream_idx + 1) % NUM_STREAMS;

        // BEGIN AI EDIT: Fix launch_gpu_kernel call with correct arguments
        launch_gpu_kernel(
            gpu_resources.get_framebuffer(), gpu_resources.get_spheres(),
            scene.spheres.size(), scene.lights.size(), tile->x_start,
            tile->y_start, tile->x_end - tile->x_start,
            tile->y_end - tile->y_start, width, height, max_depth, stream);
        // END AI EDIT

        // BEGIN AI EDIT: Download tile results from GPU to host framebuffer
        gpu_resources.download_tile(*tile, framebuffer, width, stream);
        // END AI EDIT

        tile->processed = true;
      }

      // Wait for all GPU work to complete
      for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Hybrid rendering time: " << diff.count() << " seconds"
            << std::endl;

  // Cleanup streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }

  // Verify all tiles processed
  int unprocessed = 0;
  for (const auto &tile : tiles) {
    if (!tile.processed)
      unprocessed++;
  }
  if (unprocessed > 0) {
    std::cerr << "Warning: " << unprocessed << " tiles not processed!"
              << std::endl;
  }
}

// =========================================================
// Asynchronous Pipeline Version (Advanced)
// =========================================================
void render_hybrid_pipeline(const Scene &scene, const Camera &camera,
                            std::vector<Vec3> &framebuffer, int width,
                            int height, int max_depth) {

  std::cout << "Hybrid Pipeline Rendering..." << std::endl;

  // TODO: STUDENT - Implement pipelined version
  // Stage 1: Tile generation and complexity estimation
  // Stage 2: GPU kernel execution
  // Stage 3: CPU processing
  // Stage 4: Result aggregation
  // Use pinned memory for faster transfers

  // Placeholder - calls basic hybrid version
  render_hybrid(scene, camera, framebuffer, width, height, max_depth);
}

// =========================================================
// Main Function
// =========================================================
int main(int argc, char *argv[]) {
  // Image settings
  const int width = 1280;
  const int height = 720;
  const int max_depth = 3;

  // Parse command line arguments
  // BEGIN AI EDIT: Add scene file argument support
  bool use_pipeline = false;
  int tile_size = 64;
  std::string output_file = "output_hybrid.ppm";
  std::string scene_file = "scenes/simple.txt";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--pipeline" || arg == "-p") {
      use_pipeline = true;
    } else if (arg == "--tile-size" || arg == "-t") {
      if (i + 1 < argc) {
        tile_size = std::atoi(argv[++i]);
      }
    } else if (arg == "--output" || arg == "-o") {
      if (i + 1 < argc) {
        output_file = argv[++i];
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options] [scene_file]\n";
      std::cout << "Options:\n";
      std::cout << "  --pipeline, -p        Use pipelined execution\n";
      std::cout << "  --tile-size, -t SIZE  Set tile size (default: 64)\n";
      std::cout << "  --output, -o FILE     Output filename\n";
      std::cout << "  --help, -h            Show this help message\n";
      std::cout << "\nPositional arguments:\n";
      std::cout << "  scene_file            Scene file to load (default: "
                   "scenes/simple.txt)\n";
      return 0;
    } else {
      // Assume it's a scene file
      scene_file = arg;
    }
  }
  // END AI EDIT

  // Check CUDA availability
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found. Cannot run hybrid version."
              << std::endl;
    return 1;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::cout << "Using GPU: " << props.name << std::endl;
  std::cout << "Tile size: " << tile_size << "x" << tile_size << std::endl;

  // BEGIN AI EDIT: Load scene from file using scene_loader
  std::cout << "Loading scene from: " << scene_file << std::endl;
  SceneData scene_data = load_scene(scene_file);
  Scene scene = scene_data.scene;
  // scene.print_stats();

  // Setup camera from loaded scene data
  Vec3 lookfrom = scene_data.camera.position;
  Vec3 lookat = scene_data.camera.look_at;
  Vec3 vup(0, 1, 0);
  double vfov = scene_data.camera.fov;
  // END AI EDIT

  // BEGIN AI EDIT: Fix Camera constructor call to match 3-argument signature
  Camera camera(lookfrom, lookat, vfov);
  // END AI EDIT

  // Allocate framebuffer
  std::vector<Vec3> framebuffer(width * height);

  // Render
  if (use_pipeline) {
    render_hybrid_pipeline(scene, camera, framebuffer, width, height,
                           max_depth);
  } else {
    render_hybrid(scene, camera, framebuffer, width, height, max_depth,
                  tile_size);
  }

  // Write output
  write_ppm(output_file, framebuffer, width, height);

// Performance comparison
#ifdef COMPARE_MODES
  std::cout << "\n=== Performance Comparison ===" << std::endl;

  // Run CPU-only with OpenMP
  auto start = std::chrono::high_resolution_clock::now();
  // ... CPU rendering ...
  auto end = std::chrono::high_resolution_clock::now();
  double cpu_time = std::chrono::duration<double>(end - start).count();

  // Run GPU-only
  start = std::chrono::high_resolution_clock::now();
  // ... GPU rendering ...
  end = std::chrono::high_resolution_clock::now();
  double gpu_time = std::chrono::duration<double>(end - start).count();

  // Run Hybrid
  start = std::chrono::high_resolution_clock::now();
  render_hybrid(scene, camera, framebuffer, width, height, max_depth,
                tile_size);
  end = std::chrono::high_resolution_clock::now();
  double hybrid_time = std::chrono::duration<double>(end - start).count();

  std::cout << "CPU-only time:    " << cpu_time << " seconds" << std::endl;
  std::cout << "GPU-only time:    " << gpu_time << " seconds" << std::endl;
  std::cout << "Hybrid time:      " << hybrid_time << " seconds" << std::endl;
  std::cout << "Hybrid speedup over GPU: " << gpu_time / hybrid_time << "x"
            << std::endl;
#endif

  return 0;
}
