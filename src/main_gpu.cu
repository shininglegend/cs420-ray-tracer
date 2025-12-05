// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: In Progress :)

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <math_constants.h>
#include <vector>

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constants
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
// GPU Vector and Ray Classes (simplified for CUDA)
// =========================================================

struct float3_ops {
  __device__ static float3 make(float x, float y, float z) {
    return make_float3(x, y, z);
  }

  __device__ static float3 add(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  __device__ static float3 sub(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  __device__ static float3 mul(const float3 &a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
  }

  __device__ static float3 mul(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
  }

  __device__ static float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  __device__ static float length(const float3 &v) { return sqrtf(dot(v, v)); }

  __device__ static float3 normalize(const float3 &v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
  }

  __device__ static float3 reflect(const float3 &v, const float3 &n) {
    return sub(v, mul(n, 2.0f * dot(v, n)));
  }

  __device__ static float3 lerp(const float3 &a, const float3 &b, float t) {
    return add(mul(a, 1.0f - t), mul(b, t));
  }
};

struct GPURay {
  float3 origin;
  float3 direction;

  __device__ float3 at(float t) const {
    return float3_ops::add(origin, float3_ops::mul(direction, t));
  }
};

// =========================================================
// GPU Sphere and Material Structures
// =========================================================

struct GPUMaterial {
  float3 albedo;  // Color
  float metallic; // Reflectivity
  // float roughness; // Ignored
  float shininess;
};

struct GPUSphere {
  float3 center;
  float radius;
  GPUMaterial material;

  // ===== STUDENT - IMPLEMENT GPU SPHERE INTERSECTION =====
  __device__ bool intersect(const GPURay &ray, float t_min, float t_max,
                            float &t) const {
    // From CPU version. Add float3
    // 1. Calculate discriminant
    // From Ai: Use float3_ops instead of direct operators (C++ specific)
    float3 oc = float3_ops::sub(ray.origin, center);
    double a = float3_ops::dot(ray.direction, ray.direction);
    double b = 2.0 * float3_ops::dot(oc, ray.direction);
    double c = float3_ops::dot(oc, oc) - radius * radius;

    double discriminant = b * b - 4 * a * c;

    // 2. Check if discriminant >= 0
    if (discriminant < 0) {
      // std::cout << " (negative - no intersection)\n";
      return false;
    }
    // 3. Calculate t values
    if (discriminant == 0) {
      // Only one root/intersection
      double t0 = -b / (2 * a);
      t = t0;
      return true;
    }
    // 4. Return smallest positive t (smallest root/closest intersection)
    double t1 = (-b - sqrt(discriminant)) / (2 * a);
    double t2 = (-b + sqrt(discriminant)) / (2 * a);

    // Using tmax and tmin is from Ai - I was confused as to why to use them
    // Check t1 first (smaller root)
    if (t1 >= t_min && t1 <= t_max) {
      t = t1;
      return true;
    }
    // Check t2 if t1 was out of bounds
    if (t2 >= t_min && t2 <= t_max) {
      t = t2;
      return true;
    }
    return false;
  }

  __device__ float3 normal_at(const float3 &point) const {
    return float3_ops::normalize(float3_ops::sub(point, center));
  }
};

struct GPULight {
  float3 position;
  float3 color;
  float intensity;
};

// =========================================================
// GPU Camera
// =========================================================

struct GPUCamera {
  float3 origin;
  float3 lower_left;
  float3 horizontal;
  float3 vertical;

  __device__ GPURay get_ray(float u, float v) const {
    float3 direction = float3_ops::add(
        lower_left, float3_ops::add(float3_ops::mul(horizontal, u),
                                    float3_ops::mul(vertical, v)));
    direction = float3_ops::sub(direction, origin);

    GPURay ray;
    ray.origin = origin;
    ray.direction = float3_ops::normalize(direction);
    return ray;
  }
};

// =========================================================
// TODO: STUDENT IMPLEMENTATION - GPU Ray Tracing Kernel
// =========================================================
// Implement the main ray tracing kernel that runs on the GPU.
// Each thread handles one pixel.
//
// Key differences from CPU version:
// - No recursion (use iterative approach for reflections)
// - Use shared memory for frequently accessed data
// - Be careful with memory access patterns
// =========================================================
// Consts
__constant__ float3 ambient_light = {0.1f, 0.1f, 0.1f};
__constant__ double k_specular = 0.5;

// Helper functions
__device__ bool in_shadow(const float3 &point, const GPULight &light,
                          GPUSphere *spheres, int num_spheres) {
  float3 to_light = float3_ops::sub(light.position, point);
  float light_dist = float3_ops::length(to_light);
  float3 light_dir = float3_ops::normalize(to_light);

  GPURay shadow_ray;
  shadow_ray.origin = point;
  shadow_ray.direction = light_dir;

  // Check if any sphere blocks the light
  for (int i = 0; i < num_spheres; i++) {
    float t;
    if (spheres[i].intersect(shadow_ray, EPSILON, light_dist, t)) {
      return true; // Something blocks the light
    }
  }
  return false;
}

//  c. Calculate shading (ambient + diffuse + specular)
__device__ float3 shade(const float3 &point, const float3 &normal,
                        const GPUMaterial &mat, const float3 &view_dir,
                        const float3 hit, GPUSphere *spheres, int num_spheres,
                        int sphere_idx, GPULight *lights, int num_lights) {

  // From shade function in scene.h
  // Start with ambient lighting
  float3 color =
      float3_ops::mul(spheres[sphere_idx].material.albedo, ambient_light);

  // For each light:
  for (int light_idx = 0; light_idx < num_lights; light_idx++) {
    // 1. Check if in shadow
    if (in_shadow(hit, lights[light_idx], spheres, num_spheres)) {
      // std::cout << "Skipping...";
      continue;
    }

    // 2. Calculate diffuse component (Lambert)
    float3 light_dir = float3_ops::normalize(
        float3_ops::sub(lights[light_idx].position, point));
    // BEGIN AI EDIT: Replace std::max with fmaxf for CUDA
    float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
    // END AI EDIT
    // BEGIN AI EDIT: Fix diffuse - use (1 - reflectivity) not reflectivity
    float3 diffuse = float3_ops::mul(
        float3_ops::mul(mat.albedo, (1.0f - mat.metallic)), n_dot_l);
    // END AI EDIT

    // 3. Calculate specular component (Phong)
    float3 reflect_dir =
        float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
    // BEGIN AI EDIT: Replace std::max with fmaxf for CUDA
    float r_dot_v = fmaxf(0.0f, float3_ops::dot(reflect_dir, view_dir));
    // END AI EDIT
    double spec_factor = pow(r_dot_v, mat.shininess);
    float3 specular = float3_ops::mul(
        float3_ops::mul(lights[light_idx].color, k_specular), spec_factor);

    // Add it all together
    color = float3_ops::mul(float3_ops::add(specular, diffuse), color);
    return color;
  }
}

// Render function
__global__ void render_kernel(float3 *framebuffer, GPUSphere *spheres,
                              int num_spheres, GPULight *lights, int num_lights,
                              GPUCamera camera, int width, int height,
                              int max_bounces) {

  // Calculate pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // STUDENT CODE HERE
  // Steps:
  // 1. Generate ray for this pixel
  GPURay ray = camera.get_ray(x, y);
  int pixel_idx = (y * width) + x;
  // 2. Initialize color accumulator and attenuation
  float3 color;
  float t = INFINITY;
  int sphere_idx = -1;

  // 3. Iterative ray bouncing (instead of recursion):
  for (int bounce = 0; bounce < max_bounces; bounce++) {
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
      } else {
        //  b. If no hit, add background color and break
        color = float3_ops::add(color, ambient_light);
        framebuffer[pixel_idx] = color;
        return;
      }

      float3 hit =
          float3_ops::add(ray.origin, (float3_ops::mul(ray.direction, t)));
      //  c. Calculate shading (ambient + diffuse + specular)
      // BEGIN AI EDIT: Add shade() call calculations
      float3 normal = spheres[sphere_idx].normal_at(hit);
      float3 view_dir =
          float3_ops::normalize(float3_ops::mul(ray.direction, -1.0f));
      // END AI EDIT
      color = shade(hit, normal, spheres[sphere_idx].material, view_dir, hit,
                    spheres, num_spheres, sphere_idx, lights, num_lights);

      //  d. If reflective, setup ray for next bounce
      if (spheres[sphere_idx].material.shininess > 0) {
        //  e. Accumulate color with attenuation
        // TODO:
      }
      break;
    }
  }
  // 4. Store final color in framebuffer
  framebuffer[pixel_idx] = color;
}

// =========================================================
// TODO: STUDENT OPTIMIZATION - Shared Memory Kernel
// =========================================================
// Implement an optimized version using shared memory for spheres
// that are accessed by all threads in a block.
// =========================================================

__global__ void render_kernel_optimized(float3 *framebuffer,
                                        GPUSphere *global_spheres,
                                        int num_spheres, GPULight *lights,
                                        int num_lights, GPUCamera camera,
                                        int width, int height,
                                        int max_bounces) {

  // TODO: STUDENT CODE HERE
  // 1. Declare shared memory for spheres
  //    extern __shared__ GPUSphere shared_spheres[];
  // 2. Cooperatively load spheres into shared memory
  // 3. __syncthreads()
  // 4. Use shared_spheres instead of global_spheres for intersection tests

  // For now, just call the basic kernel logic
  render_kernel(framebuffer, global_spheres, num_spheres, lights, num_lights,
                camera, width, height, max_bounces);
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
  float3 vup = make_float3(0, 1, 0);
  float vfov = 60.0f;
  float aspect = float(width) / float(height);

  // Calculate camera basis
  float theta = vfov * M_PI / 180.0f;
  float h = tanf(theta / 2.0f);
  float viewport_height = 2.0f * h;
  float viewport_width = aspect * viewport_height;
  float focal_length = 1.0f;

  float3 w = float3_ops::normalize(float3_ops::sub(lookfrom, lookat));
  float3 u = float3_ops::normalize(make_float3(vup.y * w.z - vup.z * w.y,
                                               vup.z * w.x - vup.x * w.z,
                                               vup.x * w.y - vup.y * w.x));
  float3 v = make_float3(w.y * u.z - w.z * u.y, w.z * u.x - w.x * u.z,
                         w.x * u.y - w.y * u.x);

  GPUCamera camera;
  camera.origin = lookfrom;
  camera.horizontal = float3_ops::mul(u, viewport_width);
  camera.vertical = float3_ops::mul(v, viewport_height);
  camera.lower_left = float3_ops::sub(
      float3_ops::sub(
          float3_ops::sub(lookfrom, float3_ops::mul(camera.horizontal, 0.5f)),
          float3_ops::mul(camera.vertical, 0.5f)),
      float3_ops::mul(w, focal_length));

  return camera;
}

int main(int argc, char *argv[]) {
  // Image settings
  const int width = 800;
  const int height = 600;
  const int max_bounces = 3;

  // CUDA device info
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "Using GPU: " << props.name << std::endl;
  std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
  std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock
            << " bytes\n";

  // Create scene data
  std::vector<GPUSphere> h_spheres;
  std::vector<GPULight> h_lights;

  // ===== TODO: STUDENT - CREATE GPU SCENE =====
  // Add spheres (aim for 50-100 for GPU testing)

  // Example spheres
  h_spheres.push_back({make_float3(0, 0, -20),
                       2.0f,
                       {make_float3(1, 0, 0), 0.0f, 1.0f, 10.0f}});

  h_spheres.push_back({make_float3(3, 0, -20),
                       2.0f,
                       {make_float3(0.8f, 0.8f, 0.8f), 0.8f, 0.2f, 100.0f}});

  // TODO: Add many more spheres (use loops to create patterns)

  // Lights
  h_lights.push_back({make_float3(10, 10, -10), make_float3(1, 1, 1), 0.7f});

  std::cout << "Scene: " << h_spheres.size() << " spheres, " << h_lights.size()
            << " lights\n";

  // Allocate device memory
  GPUSphere *d_spheres;
  GPULight *d_lights;
  float3 *d_framebuffer;

  CUDA_CHECK(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(GPUSphere)));
  CUDA_CHECK(cudaMalloc(&d_lights, h_lights.size() * sizeof(GPULight)));
  CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(float3)));

  // Copy scene to device
  CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(),
                        h_spheres.size() * sizeof(GPUSphere),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lights, h_lights.data(),
                        h_lights.size() * sizeof(GPULight),
                        cudaMemcpyHostToDevice));

  // Setup camera
  GPUCamera camera = setup_camera(width, height);

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

  render_kernel<<<blocks, threads>>>(d_framebuffer, d_spheres, h_spheres.size(),
                                     d_lights, h_lights.size(), camera, width,
                                     height, max_bounces);

  // For optimized version with shared memory:
  // size_t shared_size = h_spheres.size() * sizeof(GPUSphere);
  // render_kernel_optimized<<<blocks, threads, shared_size>>>(...);

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
  CUDA_CHECK(cudaFree(d_lights));
  CUDA_CHECK(cudaFree(d_framebuffer));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::cout << "Done! Output written to output_gpu.ppm\n";

  return 0;
}
