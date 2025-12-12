#include <cstddef>
#include <cuda_runtime.h>

#include "gpu_shared.h"

// Consts
__constant__ float3 ambient_light;
__constant__ double k_specular = 0.5;
__constant__ GPULight const_lights[100]; // Set max 100

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

// Calculate shading (ambient + diffuse + specular)
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

// BEGIN AI EDIT: Rename kernel, add extern "C" wrapper below
// Function for running on one tile
__global__ void gpu_kernel_impl(float3 *d_framebuffer, GPUSphere *d_spheres,
                                int num_spheres, int num_lights,
                                GPUCamera *camera, int tile_x, int tile_y,
                                int tile_width, int tile_height,
                                int image_width, int image_height,
                                int max_depth) {
  // END AI EDIT
  // STUDENT CODE HERE
  // 1. Declare shared memory for spheres
  extern __shared__ GPUSphere shared_spheres[];
  // 2. Cooperatively load spheres into shared memory
  int tid =
      threadIdx.y * blockDim.x + threadIdx.x; // Flattened thread ID (from AI
  if (tid <= num_spheres) {
    shared_spheres[tid] = d_spheres[tid];
  }

  // 3. sync
  __syncthreads();

  // Calculate pixel coordinates
  // BEGIN AI EDIT: Add tile offsets to get correct global pixel coordinates
  int x = tile_x + blockIdx.x * blockDim.x + threadIdx.x;
  int y = tile_y + blockIdx.y * blockDim.y + threadIdx.y;
  // END AI EDIT

  if (x >= image_width || y >= image_height)
    return;

  // Steps:
  // 1. Generate ray for this pixel
  float u = float(x) / float(image_width - 1);
  float v = float(y) / float(image_height - 1);
  GPURay ray = camera->get_ray(u, v);

  int pixel_idx = (y * image_width) + x;
  // BEGIN AI EDIT: Fix reflection accumulation logic
  // 2. Initialize color accumulator and attenuation
  float3 final_color = make_float3(0, 0, 0);
  float attenuation = 1.0f; // start at 1, decreases with each reflection

  // 3. Iterative ray bouncing (instead of recursion):
  for (int bounce = 0; bounce < max_depth; bounce++) {
    float t = INFINITY;
    int sphere_idx = -1;

    //  a. Find intersection
    for (int curr_sphere_idx = 0; curr_sphere_idx < num_spheres;
         curr_sphere_idx++) {
      float temp_t = 0;
      // If an intersection, and smallest t, save it.
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
  d_framebuffer[pixel_idx] = final_color;
  return;
}

// BEGIN AI EDIT: Add extern "C" wrappers to be called from main_hybrid.cpp
extern "C" void launch_gpu_kernel(float3 *d_framebuffer, GPUSphere *d_spheres,
                                  int num_spheres, int num_lights,
                                  GPUCamera *camera, int tile_x, int tile_y,
                                  int tile_width, int tile_height,
                                  int image_width, int image_height,
                                  int max_depth, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((tile_width + block.x - 1) / block.x,
            (tile_height + block.y - 1) / block.y);
  // BEGIN AI EDIT: Pass shared memory size for spheres
  size_t shared_mem_size = num_spheres * sizeof(GPUSphere);
  gpu_kernel_impl<<<grid, block, shared_mem_size, stream>>>(
      // END AI EDIT
      d_framebuffer, d_spheres, num_spheres, num_lights, camera, tile_x, tile_y,
      tile_width, tile_height, image_width, image_height, max_depth);
}

extern "C" void upload_lights_and_ambience(GPULight *lights, int count,
                                           float3 ambience) {
  CUDA_CHECK(
      cudaMemcpyToSymbol(const_lights, lights, count * sizeof(GPULight)));
  CUDA_CHECK(cudaMemcpyToSymbol(ambient_light, &ambience, sizeof(float3)));
}
// END AI EDIT