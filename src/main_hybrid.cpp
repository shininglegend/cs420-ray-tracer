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

const int NUM_STREAMS = 3;

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
                                  int num_spheres, int num_lights,
                                  GPUCamera *camera, int tile_x, int tile_y,
                                  int tile_width, int tile_height,
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
extern "C" void upload_lights_and_ambience(GPULight *lights, int coun,
                                           float3 ambient_light);

class GPUResources {
private:
  float3 *d_framebuffer;
  GPUSphere *d_spheres;
  GPUCamera *d_camera;
  size_t fb_size;
  size_t spheres_size;

public:
  GPUResources(int width, int height, int num_spheres, int num_lights) {
    fb_size = width * height * sizeof(float3);
    spheres_size = num_spheres * sizeof(GPUSphere); // Switch to object

    CUDA_CHECK(cudaMalloc(&d_framebuffer, fb_size));
    CUDA_CHECK(cudaMalloc(&d_spheres, spheres_size));
    CUDA_CHECK(cudaMalloc(&d_camera, sizeof(GPUCamera)));
  }

  ~GPUResources() {
    CUDA_CHECK(cudaFree(d_framebuffer));
    CUDA_CHECK(cudaFree(d_spheres));
  }

  void upload_scene(const Scene &scene, int width, int height) {
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
    // AI EDIT: use make_float3 instead of float3 constructor
    float3 ambient_light = make_float3(
        scene.ambient_light.x, scene.ambient_light.y, scene.ambient_light.z);
    // END AI EDIT
    upload_lights_and_ambience(h_lights.data(), h_lights.size(), ambient_light);

    // Add the camera
    GPUCamera camera;
    if (scene.has_camera) {
      // Use camera from scene file
      float3 lookfrom =
          make_float3(scene.camera.position.x, scene.camera.position.y,
                      scene.camera.position.z);
      float3 lookat =
          make_float3(scene.camera.look_at.x, scene.camera.look_at.y,
                      scene.camera.look_at.z);
      float vfov = scene.camera.fov;

      // AI EDIT: Match CPU camera setup exactly
      camera.origin = lookfrom;
      camera.fov = vfov;

      // Calculate basis vectors like CPU version
      camera.forward = float3_ops::normalize(float3_ops::sub(lookat, lookfrom));
      float3 world_up = make_float3(0, 1, 0);

      // right = cross(forward, world_up)
      camera.right = float3_ops::normalize(make_float3(
          camera.forward.y * world_up.z - camera.forward.z * world_up.y,
          camera.forward.z * world_up.x - camera.forward.x * world_up.z,
          camera.forward.x * world_up.y - camera.forward.y * world_up.x));

      // up = cross(right, forward)
      camera.up = make_float3(
          camera.right.y * camera.forward.z - camera.right.z * camera.forward.y,
          camera.right.z * camera.forward.x - camera.right.x * camera.forward.z,
          camera.right.x * camera.forward.y -
              camera.right.y * camera.forward.x);
      // END AI EDIT
    } else {
      // Fallback to default camera
      camera = setup_camera(width, height);
    }
    // AI EDIT: fix cudaMemcpy to include all required parameters
    CUDA_CHECK(cudaMemcpy(d_camera, &camera, sizeof(GPUCamera),
                          cudaMemcpyHostToDevice));
    // END AI EDIT
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
  GPUCamera *get_camera() { return d_camera; }
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

  // My Idea 1: Sample an X (top left, top right, middle, bottom left, bottom
  // right) Count number of intersections. Middle ray counts each intersection
  // 2x. BEGIN AI EDIT: Implement X-pattern sampling for complexity estimation
  int complexity = 0;

  // Sample points: corners + middle (X pattern)
  int sample_x[5] = {tile.x_start, tile.x_end - 1,
                     (tile.x_start + tile.x_end) / 2, tile.x_start,
                     tile.x_end - 1};
  int sample_y[5] = {tile.y_start, tile.y_start,
                     (tile.y_start + tile.y_end) / 2, tile.y_end - 1,
                     tile.y_end - 1};
  int weight[5] = {1, 1, 2, 1, 1}; // Middle ray counts 2x

  for (int i = 0; i < 5; i++) {
    double u = double(sample_x[i]) / 1280.0; // Assuming image width
    double v = double(sample_y[i]) / 720.0;  // Assuming image height
    Ray ray = camera.get_ray(u, v);

    // Count intersections with all spheres
    for (const auto &sphere : scene.spheres) {
      double t;
      if (sphere.intersect(ray, t)) {
        complexity += weight[i];
      }
    }
  }
  // END AI EDIT

  return complexity; // Base cost + intersection complexity
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

  // Split tiles between CPU and GPU based on complexity
  std::queue<Tile *> cpu_queue;
  std::queue<Tile *> gpu_queue;
  std::vector<Tile> tiles;
  cudaStream_t streams[NUM_STREAMS];
  // Create tiles
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
  gpu_resources.upload_scene(scene, width, height);

  // Create CUDA streams for pipelining
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // TODO: STUDENT - Implement work distribution

  // Simple strategy: complex tiles to CPU, simple to GPU
  // Threshold if at least x of the rays hit
  // I'm aiming for about 70% on GPU, 30% on CPU to start
  int complexity_threshold = 7;

  for (auto &tile : tiles) {
    if (tile.complexity_estimate > complexity_threshold) {
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

        // STUDENT - Launch GPU kernel for this tile
        // Use streams for asynchronous execution
        cudaStream_t stream = streams[stream_idx];
        stream_idx = (stream_idx + 1) % NUM_STREAMS;

        // BEGIN AI EDIT: Fix launch_gpu_kernel call with correct arguments
        launch_gpu_kernel(
            gpu_resources.get_framebuffer(), gpu_resources.get_spheres(),
            scene.spheres.size(), scene.lights.size(),
            gpu_resources.get_camera(), tile->x_start, tile->y_start,
            tile->x_end - tile->x_start, tile->y_end - tile->y_start, width,
            height, max_depth, stream);
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
                            int height, int max_depth, int tile_size = 64) {

  std::cout << "Hybrid Pipeline Rendering..." << std::endl;

  // BEGIN AI EDIT: Improved pseudocode for pipelined version
  // TODO: STUDENT - Implement pipelined version
  //
  // Key insight: The bottleneck is memory transfer overhead (thousands of
  // small cudaMemcpy calls). Solution: Use pinned memory + single bulk
  // transfer.
  //
  // ==================== SETUP ====================
  // 1. Allocate PINNED host memory for entire framebuffer:
  float3 *h_pinned_fb;
  CUDA_CHECK(cudaMallocHost(&h_pinned_fb, width * height * sizeof(float3)));

  // 2. Create tiles and estimate complexity (reuse existing code)
  // Split tiles between CPU and GPU based on complexity
  std::queue<Tile *> cpu_queue;
  std::queue<Tile *> gpu_queue;
  std::vector<Tile> tiles;
  cudaStream_t streams[NUM_STREAMS];
  // Create tiles
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
  gpu_resources.upload_scene(scene, width, height);

  // Create CUDA streams for pipelining
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // TODO: STUDENT - Implement work distribution

  // Simple strategy: complex tiles to CPU, simple to GPU
  // Threshold if at least x of the rays hit
  // I'm aiming for about 70% on GPU, 30% on CPU to start
  int complexity_threshold = 9;

  for (auto &tile : tiles) {
    if (tile.complexity_estimate > complexity_threshold) {
      cpu_queue.push(&tile);
    } else {
      gpu_queue.push(&tile);
    }
  }

  std::cout << "Distribution: " << cpu_queue.size() << " tiles to CPU, "
            << gpu_queue.size() << " tiles to GPU" << std::endl;
  // Track the tiles the GPU processed
  std::vector<Tile *> gpu_tiles_processed;

  auto start = std::chrono::high_resolution_clock::now();

  // ==================== EXECUTION ====================
  // 4. Launch CPU and GPU work in parallel using OpenMP sections:

#pragma omp parallel sections
  {
#pragma omp section // GPU SECTION
    {
      // Launch ALL GPU tile kernels (no sync between launches)
      // launch_gpu_kernel(..., tile, stream[i++ % NUM_STREAMS]);
      int i = 0;
      while (!gpu_queue.empty()) {
        Tile *tile = gpu_queue.front();
        gpu_tiles_processed.push_back(tile);
        gpu_queue.pop();
        launch_gpu_kernel(
            gpu_resources.get_framebuffer(), gpu_resources.get_spheres(),
            scene.spheres.size(), scene.lights.size(),
            gpu_resources.get_camera(), tile->x_start, tile->y_start,
            tile->x_end - tile->x_start, tile->y_end - tile->y_start, width,
            height, max_depth, streams[i++ % NUM_STREAMS]);
        tile->processed = true;
        i++;
      }

      // Single sync point after all launches
      cudaDeviceSynchronize();

      // ONE bulk download of entire GPU framebuffer (Bug fix from Ai here)
      cudaMemcpy(h_pinned_fb, gpu_resources.get_framebuffer(),
                 width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    }

#pragma omp section // CPU SECTION
    {
      // BEGIN AI EDIT: Copy queue to vector for OpenMP parallel for
      std::vector<Tile *> cpu_tiles_vec;
      while (!cpu_queue.empty()) {
        cpu_tiles_vec.push_back(cpu_queue.front());
        cpu_queue.pop();
      }
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < cpu_tiles_vec.size(); i++) {
        Tile *tile = cpu_tiles_vec[i];
        process_tile_cpu(*tile, scene, camera, framebuffer, width, height,
                         max_depth);
        tile->processed = true;
      }
      // END AI EDIT
    }
  }

  // ==================== MERGE ====================
  // 5. Convert GPU results (float3) to final framebuffer (Vec3):
  //    - Only convert pixels from GPU tiles (CPU tiles already in framebuffer)
  // Loop over GPU tiles
  // BEGIN AI EDIT: Fix range-based for loop syntax
  for (Tile *tile : gpu_tiles_processed) {
    for (int x = tile->x_start; x < tile->x_end; x++) {
      for (int y = tile->y_start; y < tile->y_end; y++) {
        framebuffer[y * width + x] =
            Vec3(h_pinned_fb[y * width + x].x, h_pinned_fb[y * width + x].y,
                 h_pinned_fb[y * width + x].z);
      }
    }
  }
  // END AI EDIT

  // ==================== CLEANUP ====================
  CUDA_CHECK(cudaFreeHost(h_pinned_fb));

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
  // Expected speedup: Eliminates ~13,000 cudaMemcpy calls → 1 bulk transfer
  // END Psedeudocode Ai Edit
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
  Scene scene = load_scene(scene_file);
  // scene.print_stats();

  // Setup camera from loaded scene data
  Vec3 lookfrom = scene.camera.position;
  Vec3 lookat = scene.camera.look_at;
  Vec3 vup(0, 1, 0);
  double vfov = scene.camera.fov;
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
