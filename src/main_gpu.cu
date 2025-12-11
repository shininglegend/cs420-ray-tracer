// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: In Progress :)

#include "scene_loader.h"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <math_constants.h>
#include <vector>

#include "gpu_shared.h"

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// // Constants
#ifndef EPSILON
#define EPSILON 0.001
#endif

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
// STUDENT IMPLEMENTATION - GPU Ray Tracing Kernel
// =========================================================
// Implement the main ray tracing kernel that runs on the GPU.
// Each thread handles one pixel.
//
// Key differences from CPU version:
// - No recursion (use iterative approach for reflections)
// - Use shared memory for frequently accessed data
// - Be careful with memory access patterns
// =========================================================

// Render function
__global__ void render_kernel(float3 *framebuffer, GPUSphere *spheres,
                              int num_spheres, int num_lights, GPUCamera camera,
                              int width, int height, int max_bounces) {

  // Calculate pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // STUDENT CODE HERE
  // Steps:
  // 1. Generate ray for this pixel
  // BEGIN AI EDIT: Fix UV coordinates for proper camera ray generation
  float u = float(x) / float(width - 1);
  float v = float(y) / float(height - 1);
  GPURay ray = camera.get_ray(u, v);
  // END AI EDIT
  int pixel_idx = (y * width) + x;
  // 2. Initialize color accumulator and attenuation
  float3 final_color = make_float3(0, 0, 0);
  float attenuation = 1.0f; // start at 1, decreases with each reflection

  // 3. Iterative ray bouncing (instead of recursion):
  for (int bounce = 0; bounce < max_bounces; bounce++) {
    float t = INFINITY;
    int sphere_idx = -1;

    //  a. Find intersection
    for (int curr_sphere_idx = 0; curr_sphere_idx < num_spheres;
         curr_sphere_idx++) {
      float temp_t = 0;
      // If an intersection, and smallest t, save it.
      // TODO: are t_min and t_max set right?
      if (spheres[curr_sphere_idx].intersect(ray, EPSILON, INFINITY, temp_t)) {
        if (temp_t < t) {
          sphere_idx = curr_sphere_idx;
          t = temp_t;
        }
      }
    }

    //  b. If no hit, add background color and break
    if (sphere_idx == -1) {
      // BEGIN AI EDIT: Use sky gradient like CPU version
      float t_sky = 0.5f * (ray.direction.y + 1.0f);
      float3 white = make_float3(1.0f, 1.0f, 1.0f);
      float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
      float3 sky_color = float3_ops::lerp(white, sky_blue, t_sky);
      final_color =
          float3_ops::add(final_color, float3_ops::mul(sky_color, attenuation));
      // END AI EDIT
      break;
    }

    // AI EDIT: Fix hit point calculation
    float3 hit = ray.at(t);
    //  c. Calculate shading (ambient + diffuse + specular)
    float3 normal = spheres[sphere_idx].normal_at(hit);
    float3 view_dir =
        float3_ops::normalize(float3_ops::mul(ray.direction, -1.0f));
    float3 color = shade(hit, normal, spheres[sphere_idx].material, view_dir,
                         hit, spheres, num_spheres, sphere_idx, num_lights);

    //  d. Accumulate color weighted by attenuation and reflectivity
    // AI EDIT: Fix reflection accumulation logic to calculate attenuation
    float reflectivity = spheres[sphere_idx].material.metallic;
    final_color = float3_ops::add(
        final_color,
        float3_ops::mul(color, attenuation * (1.0f - reflectivity)));

    // If not reflective or attenuation neligible, end
    if (reflectivity <= 0.0f || attenuation < 0.0f) {
      break;
    }

    //  e. If reflective, setup ray for next bounce
    float3 reflected_dir = float3_ops::reflect(ray.direction, normal);
    ray.origin = float3_ops::add(hit, float3_ops::mul(normal, EPSILON));
    ray.direction = reflected_dir;
    // Update attenuation for next bounce
    attenuation *= reflectivity;
  }
  // 4. Store final color in framebuffer
  framebuffer[pixel_idx] = final_color;
}

// =========================================================
// STUDENT OPTIMIZATION - Shared Memory Kernel
// =========================================================
// Implement an optimized version using shared memory for spheres
// that are accessed by all threads in a block.
// =========================================================

__global__ void render_kernel_optimized(float3 *framebuffer,
                                        GPUSphere *global_spheres,
                                        int num_spheres, int num_lights,
                                        GPUCamera camera, int width, int height,
                                        int max_bounces) {

  // STUDENT CODE HERE
  // 1. Declare shared memory for spheres
  extern __shared__ GPUSphere shared_spheres[];
  // 2. Cooperatively load spheres into shared memory
  int tid =
      threadIdx.y * blockDim.x + threadIdx.x; // Flattened thread ID (from AI
  if (tid <= num_spheres) {
    shared_spheres[tid] = global_spheres[tid];
  }

  // 3. sync
  __syncthreads();

  // 4. Use shared_spheres instead of global_spheres for intersection tests
  // NOTE: Code copied from above and edited

  // Calculate pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Steps:
  // 1. Generate ray for this pixel
  // AI EDIT: Fix UV coordinates for proper camera ray generation
  float u = float(x) / float(width - 1);
  float v = float(y) / float(height - 1);
  GPURay ray = camera.get_ray(u, v);

  int pixel_idx = (y * width) + x;
  // BEGIN AI EDIT: Fix reflection accumulation logic
  // 2. Initialize color accumulator and attenuation
  float3 final_color = make_float3(0, 0, 0);
  float attenuation = 1.0f; // start at 1, decreases with each reflection

  // 3. Iterative ray bouncing (instead of recursion):
  for (int bounce = 0; bounce < max_bounces; bounce++) {
    float t = INFINITY;
    int sphere_idx = -1;

    //  a. Find intersection
    for (int curr_sphere_idx = 0; curr_sphere_idx < num_spheres;
         curr_sphere_idx++) {
      float temp_t = 0;
      // If an intersection, and smallest t, save it.
      // TODO: are t_min and t_max set right?
      if (shared_spheres[curr_sphere_idx].intersect(ray, EPSILON, INFINITY,
                                                    temp_t)) {
        if (temp_t < t) {
          sphere_idx = curr_sphere_idx;
          t = temp_t;
        }
      }
    }

    //  b. If no hit, add background color and break
    if (sphere_idx == -1) {
      // BEGIN AI EDIT: Use sky gradient like CPU version
      float t_sky = 0.5f * (ray.direction.y + 1.0f);
      float3 white = make_float3(1.0f, 1.0f, 1.0f);
      float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
      float3 sky_color = float3_ops::lerp(white, sky_blue, t_sky);
      final_color =
          float3_ops::add(final_color, float3_ops::mul(sky_color, attenuation));
      // END AI EDIT
      break;
    }

    // AI EDIT: Fix hit point calculation
    float3 hit = ray.at(t);
    //  c. Calculate shading (ambient + diffuse + specular)
    float3 normal = shared_spheres[sphere_idx].normal_at(hit);
    float3 view_dir =
        float3_ops::normalize(float3_ops::mul(ray.direction, -1.0f));
    float3 color =
        shade(hit, normal, shared_spheres[sphere_idx].material, view_dir, hit,
              shared_spheres, num_spheres, sphere_idx, num_lights);

    //  d. Accumulate color weighted by attenuation and reflectivity
    float reflectivity = shared_spheres[sphere_idx].material.metallic;
    final_color = float3_ops::add(
        final_color,
        float3_ops::mul(color, attenuation * (1.0f - reflectivity)));

    // If not reflective, end
    if (reflectivity <= 0.0f) {
      break;
    }

    //  e. If reflective, setup ray for next bounce
    float3 reflected_dir = float3_ops::reflect(ray.direction, normal);
    ray.origin = float3_ops::add(hit, float3_ops::mul(normal, EPSILON));
    ray.direction = reflected_dir;
    // Update attenuation for next bounce
    attenuation *= reflectivity;
    // END AI EDIT
  }
  // 4. Store final color in framebuffer
  framebuffer[pixel_idx] = final_color;
}

// =========================================================
// Host Functions
// =========================================================

void write_ppm(const std::string &filename,
               const std::vector<float3> &framebuffer, int width, int height) {
  std::ofstream file(filename);
  file << "P3\n" << width << " " << height << "\n255\n";

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      float3 color = framebuffer[j * width + i];
      int r = int(255.99f * fminf(1.0f, color.x));
      int g = int(255.99f * fminf(1.0f, color.y));
      int b = int(255.99f * fminf(1.0f, color.z));
      file << r << " " << g << " " << b << "\n";
    }
  }
}

GPUCamera setup_camera(int width, int height) {
  // Camera parameters
  float3 lookfrom = make_float3(0, 2, 5);
  float3 lookat = make_float3(0, 0, -20);
  float vfov = 60.0f;

  // AI EDIT: Match CPU camera setup exactly
  GPUCamera camera;
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
      camera.right.x * camera.forward.y - camera.right.y * camera.forward.x);
  // END AI EDIT

  return camera;
}

int main(int argc, char *argv[]) {
  // Image settings
  const int width = 640;
  const int height = 480;
  const int max_bounces = 10;

  // BEGIN AI EDIT: Parse command-line arguments for scene file
  std::string scene_file = "scenes/simple.txt";
  if (argc > 1) {
    scene_file = argv[1];
  }
  // END AI EDIT

  // CUDA device info
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "Using GPU: " << props.name << std::endl;
  std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
  std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock
            << " bytes\n";

  // BEGIN AI EDIT: Load scene from file and convert to GPU structures
  std::cout << "Loading scene from: " << scene_file << "\n\n";
  SceneData scene_data = load_scene(scene_file);
  // print_scene_info(scene_data);

  // Convert CPU scene to GPU structures
  std::vector<GPUSphere> h_spheres;
  std::vector<GPULight> h_lights;

  // Convert spheres
  for (const auto &sphere : scene_data.scene.spheres) {
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

  // Convert lights
  if (scene_data.scene.lights.size() > 100) {
    std::cerr << "Error: Scene has " << scene_data.scene.lights.size()
              << " lights, but maximum is 100\n";
    return 1;
  }

  for (const auto &light : scene_data.scene.lights) {
    GPULight gpu_light;
    gpu_light.position =
        make_float3(light.position.x, light.position.y, light.position.z);
    gpu_light.color = make_float3(light.color.x, light.color.y, light.color.z);
    gpu_light.intensity = light.intensity;
    h_lights.push_back(gpu_light);
  }
  // END AI EDIT

  std::cout << "Scene: " << h_spheres.size() << " spheres, " << h_lights.size()
            << " lights\n";

  // Allocate device memory
  GPUSphere *d_spheres;
  float3 *d_framebuffer;

  CUDA_CHECK(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(GPUSphere)));
  CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(float3)));

  // Copy scene to device
  CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(),
                        h_spheres.size() * sizeof(GPUSphere),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(const_lights, h_lights.data(),
                                h_lights.size() * sizeof(GPULight)));

  // BEGIN AI EDIT: Setup camera from loaded scene data
  GPUCamera camera;
  if (scene_data.has_camera) {
    // Use camera from scene file
    float3 lookfrom =
        make_float3(scene_data.camera.position.x, scene_data.camera.position.y,
                    scene_data.camera.position.z);
    float3 lookat =
        make_float3(scene_data.camera.look_at.x, scene_data.camera.look_at.y,
                    scene_data.camera.look_at.z);
    float vfov = scene_data.camera.fov;

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
        camera.right.x * camera.forward.y - camera.right.y * camera.forward.x);
    // END AI EDIT
  } else {
    // Fallback to default camera
    camera = setup_camera(width, height);
  }
  // END AI EDIT

  // Configure kernel launch
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

  std::cout << "Launching kernel with " << blocks.x << "x" << blocks.y
            << " blocks of " << threads.x << "x" << threads.y << " threads\n";

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Render
  std::cout << "Rendering..." << std::endl;
  CUDA_CHECK(cudaEventRecord(start));

  // ===== TODO: STUDENT - CHOOSE KERNEL VERSION =====
  // Start with basic kernel, then implement and test optimized version

  // render_kernel<<<blocks, threads>>>(d_framebuffer, d_spheres,
  // h_spheres.size(),
  //                                    d_lights, h_lights.size(), camera,
  //                                    width, height, max_bounces);

  // For optimized version with shared memory:
  size_t shared_size = h_spheres.size() * sizeof(GPUSphere);
  render_kernel_optimized<<<blocks, threads, shared_size>>>(
      d_framebuffer, d_spheres, h_spheres.size(), h_lights.size(), camera,
      width, height, max_bounces);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "GPU rendering time: " << milliseconds / 1000.0f << " seconds\n";

  // Copy result back to host
  std::vector<float3> h_framebuffer(width * height);
  CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer,
                        width * height * sizeof(float3),
                        cudaMemcpyDeviceToHost));

  // Write output
  write_ppm("output_gpu.ppm", h_framebuffer, width, height);

  // Cleanup
  CUDA_CHECK(cudaFree(d_spheres));
  // CUDA_CHECK(cudaFree(d_lights));
  CUDA_CHECK(cudaFree(d_framebuffer));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::cout << "Done! Output written to output_gpu.ppm\n";

  return 0;
}
