// This includes shared functions between the hybrid and GPU versions.
// TODO: Add the other functions. It will have errors because cuda runtime can't be found.
#include "scene_loader.h"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <math_constants.h>
#include <vector>
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

//  c. Calculate shading (ambient + diffuse + specular)
__device__ float3 shade(const float3 &point, const float3 &normal,
                        const GPUMaterial &mat, const float3 &view_dir,
                        const float3 hit, GPUSphere *spheres, int num_spheres,
                        int sphere_idx, int num_lights) {

  // From shade function in scene.h
  // Start with ambient lighting
  float3 color =
      float3_ops::mul(spheres[sphere_idx].material.albedo, ambient_light);

  // For each light:
  for (int light_idx = 0; light_idx < num_lights; light_idx++) {
    // 1. Check if in shadow
    if (in_shadow(hit, const_lights[light_idx], spheres, num_spheres)) {
      // std::cout << "Skipping...";
      continue;
    }

    // 2. Calculate diffuse component (Lambert)
    float3 light_dir = float3_ops::normalize(
        float3_ops::sub(const_lights[light_idx].position, point));
    // AI EDIT: Replace std::max with fmaxf for CUDA
    float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
    // AI EDIT: Fix diffuse - use (1 - reflectivity) not reflectivity
    float3 diffuse = float3_ops::mul(
        float3_ops::mul(mat.albedo, (1.0f - mat.metallic)), n_dot_l);

    // 3. Calculate specular component (Phong)
    float3 reflect_dir =
        float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
    // AI EDIT: Replace std::max with fmaxf for CUDA
    float r_dot_v = fmaxf(0.0f, float3_ops::dot(reflect_dir, view_dir));
    double spec_factor = pow(r_dot_v, mat.shininess);
    float3 specular = float3_ops::mul(
        float3_ops::mul(const_lights[light_idx].color, k_specular),
        spec_factor);

    color = float3_ops::add(color, float3_ops::add(diffuse, specular));
  }
  return color;
}