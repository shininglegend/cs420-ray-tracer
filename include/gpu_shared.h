// This includes shared functions between the hybrid and GPU versions.
// TODO: Add the other functions. It will have errors because cuda runtime can't
// be found.
#include "scene_loader.h"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <math_constants.h>
#include <vector>

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
  __host__ __device__ static float3 make(float x, float y, float z) {
    return make_float3(x, y, z);
  }

  __host__ __device__ static float3 add(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  __host__ __device__ static float3 sub(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  __host__ __device__ static float3 mul(const float3 &a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
  }

  __host__ __device__ static float3 mul(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
  }

  __host__ __device__ static float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  __host__ __device__ static float length(const float3 &v) {
    return sqrtf(dot(v, v));
  }

  __host__ __device__ static float3 normalize(const float3 &v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
  }

  __host__ __device__ static float3 reflect(const float3 &v, const float3 &n) {
    return sub(v, mul(n, 2.0f * dot(v, n)));
  }

  __host__ __device__ static float3 lerp(const float3 &a, const float3 &b,
                                         float t) {
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
    float a = float3_ops::dot(ray.direction, ray.direction);
    float b = 2.0 * float3_ops::dot(oc, ray.direction);
    float c = float3_ops::dot(oc, oc) - radius * radius;

    float discriminant = b * b - 4 * a * c;

    // 2. Check if discriminant >= 0
    if (discriminant < 0) {
      // std::cout << " (negative - no intersection)\n";
      return false;
    }
    // 3. Calculate t values
    if (discriminant == 0) {
      // Only one root/intersection
      float t0 = -b / (2 * a);
      t = t0;
      return true;
    }
    // 4. Return smallest positive t (smallest root/closest intersection)
    float t1 = (-b - sqrt(discriminant)) / (2 * a);
    float t2 = (-b + sqrt(discriminant)) / (2 * a);

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
  // AI EDIT: Add fields to match CPU camera implementation
  float3 forward;
  float3 right;
  float3 up;
  float fov;
  // END AI EDIT

  __device__ GPURay get_ray(float u, float v) const {
    // AI EDIT: Match CPU camera ray generation exactly
    float aspect = 1.0f;
    float scale = tanf(fov * 0.5f * M_PI / 180.0f);

    float3 direction = float3_ops::add(
        forward,
        float3_ops::add(float3_ops::mul(right, (u - 0.5f) * scale * aspect),
                        float3_ops::mul(up, (v - 0.5f) * scale)));
    // END AI EDIT

    GPURay ray;
    ray.origin = origin;
    ray.direction = float3_ops::normalize(direction);
    return ray;
  }
};

// AI EDIT: add inline to prevent multiple definition linker error
inline GPUCamera setup_camera(int width, int height) {
// END AI EDIT
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