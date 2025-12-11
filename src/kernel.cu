// TODO: import from the gpu_shared.h header file
#include "gpu_shared.h"



__global__ void launch_gpu_kernel(float3 *d_framebuffer, GPUSphere *d_spheres,
                                  int num_spheres, GPULight *d_lights,
                                  int num_lights, float *camera_params,
                                  int tile_x, int tile_y, int tile_width,
                                  int tile_height, int image_width,
                                  int image_height, int max_depth,
                                  cudaStream_t stream) {
  // // STUDENT CODE HERE
  // // 1. Declare shared memory for spheres
  // extern __shared__ GPUSphere shared_spheres[];
  // // 2. Cooperatively load spheres into shared memory
  // int tid =
  //     threadIdx.y * blockDim.x + threadIdx.x; // Flattened thread ID (from AI
  // if (tid <= num_spheres) {
  //   shared_spheres[tid] = d_spheres[tid];
  // }

  // // 3. sync
  // __syncthreads();

  // // 4. Use shared_spheres instead of global_spheres for intersection tests
  // // NOTE: Code copied from above and edited

  // // Calculate pixel coordinates
  // int x = blockIdx.x * blockDim.x + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;

  // if (x >= image_width || y >= image_height)
  //   return;

  // // Steps:
  // // 1. Generate ray for this pixel
  // float u = float(x) / float(image_width - 1);
  // float v = float(y) / float(image_height - 1);
  // GPURay ray = camera.get_ray(u, v);

  // int pixel_idx = (y * width) + x;
  // // BEGIN AI EDIT: Fix reflection accumulation logic
  // // 2. Initialize color accumulator and attenuation
  // float3 final_color = make_float3(0, 0, 0);
  // float attenuation = 1.0f; // start at 1, decreases with each reflection

  // // 3. Iterative ray bouncing (instead of recursion):
  // for (int bounce = 0; bounce < max_bounces; bounce++) {
  //   float t = INFINITY;
  //   int sphere_idx = -1;

  //   //  a. Find intersection
  //   for (int curr_sphere_idx = 0; curr_sphere_idx < num_spheres;
  //        curr_sphere_idx++) {
  //     float temp_t = 0;
  //     // If an intersection, and smallest t, save it.
  //     // TODO: are t_min and t_max set right?
  //     if (shared_spheres[curr_sphere_idx].intersect(ray, EPSILON, INFINITY,
  //                                                   temp_t)) {
  //       if (temp_t < t) {
  //         sphere_idx = curr_sphere_idx;
  //         t = temp_t;
  //       }
  //     }
  //   }

  //   //  b. If no hit, add background color and break
  //   if (sphere_idx == -1) {
  //     // BEGIN AI EDIT: Use sky gradient like CPU version
  //     float t_sky = 0.5f * (ray.direction.y + 1.0f);
  //     float3 white = make_float3(1.0f, 1.0f, 1.0f);
  //     float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
  //     float3 sky_color = float3_ops::lerp(white, sky_blue, t_sky);
  //     final_color =
  //         float3_ops::add(final_color, float3_ops::mul(sky_color, attenuation));
  //     // END AI EDIT
  //     break;
  //   }

  //   // AI EDIT: Fix hit point calculation
  //   float3 hit = ray.at(t);
  //   //  c. Calculate shading (ambient + diffuse + specular)
  //   float3 normal = shared_spheres[sphere_idx].normal_at(hit);
  //   float3 view_dir =
  //       float3_ops::normalize(float3_ops::mul(ray.direction, -1.0f));
  //   float3 color =
  //       shade(hit, normal, shared_spheres[sphere_idx].material, view_dir, hit,
  //             shared_spheres, num_spheres, sphere_idx, num_lights);

  //   //  d. Accumulate color weighted by attenuation and reflectivity
  //   float reflectivity = shared_spheres[sphere_idx].material.metallic;
  //   final_color = float3_ops::add(
  //       final_color,
  //       float3_ops::mul(color, attenuation * (1.0f - reflectivity)));

  //   // If not reflective, end
  //   if (reflectivity <= 0.0f) {
  //     break;
  //   }

  //   //  e. If reflective, setup ray for next bounce
  //   float3 reflected_dir = float3_ops::reflect(ray.direction, normal);
  //   ray.origin = float3_ops::add(hit, float3_ops::mul(normal, EPSILON));
  //   ray.direction = reflected_dir;
  //   // Update attenuation for next bounce
  //   attenuation *= reflectivity;
  //   // END AI EDIT
  // }
  // // 4. Store final color in framebuffer
  // framebuffer[pixel_idx] = final_color;
  // return;
}