#include "camera.h"
#include "ray.h"
#include "scene.h"
#include "scene_loader.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <ray_math_constants.h>
#include <string>
#include <vector>

// Trace a single ray through the scene
Vec3 trace_ray(const Ray &ray, const Scene &scene, int depth) {
  if (depth <= 0)
    return Vec3(0, 0, 0);

  // STUDENT IMPLEMENTATION (4)
  // YOUR CODE HERE
  double t;
  int sphere_idx;

  // 1. Calculate hit point (scene.h.intersection)
  if (!scene.find_intersection(ray, t, sphere_idx)) {
    // Sky color gradient
    double t = 0.5 * (ray.direction.y + 1.0);
    return Vec3(1, 1, 1) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
  }
  // Find the actual hitpoint
  Vec3 hit = ray.origin + ray.direction * t;

  // Normalize
  Vec3 norm = scene.spheres[sphere_idx].normal_at(hit);

  // 2. Call scene.shade() for color
  Vec3 view_dir = (ray.origin - hit).normalized(); // From Ai
  Vec3 shade =
      scene.shade(hit, norm, scene.spheres[sphere_idx].material, view_dir);

  // 3. If material is reflective, recursively trace reflection ray
  if (scene.spheres[sphere_idx].material.reflectivity > 0) {
    // BEGIN AI EDIT: Implement recursive reflection
    Vec3 reflected_dir = ray.direction - norm * 2.0 * dot(ray.direction, norm);
    // BEGIN AI EDIT: Offset reflection ray origin to avoid self-intersection
    // (fixes speckling)
    Ray reflected_ray(hit + norm * EPSILON, reflected_dir);
    // END AI EDIT
    Vec3 reflected_color = trace_ray(reflected_ray, scene, depth - 1);

    double refl = scene.spheres[sphere_idx].material.reflectivity;
    shade = shade * (1.0 - refl) + reflected_color * refl;
    // END AI EDIT
  }

  return shade;
}

// Adjust for gamma 2 (unused)
inline double linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return std::sqrt(linear_component);

  return 0;
}

// Write image to PPM file
void write_ppm(const std::string &filename,
               const std::vector<Vec3> &framebuffer, int width, int height) {
  std::ofstream file(filename);
  file << "P3\n" << width << " " << height << "\n255\n";

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      Vec3 color = framebuffer[j * width + i];
      auto r = color.x;
      auto g = color.y;
      auto b = color.z;
      // BEGIN AI EDIT: Remove gamma correction to match GPU version
      // r = linear_to_gamma(r);
      // g = linear_to_gamma(g);
      // b = linear_to_gamma(b);
      // END AI EDIT
      r = int(255.99 * std::min(1.0, r));
      g = int(255.99 * std::min(1.0, g));
      b = int(255.99 * std::min(1.0, b));
      file << r << " " << g << " " << b << "\n";
    }
  }
}

int main(int argc, char *argv[]) {
  // Image settings
  const int width = 1280;
  const int height = 720;
  const int max_depth = 10;

  // BEGIN EDIT: Parse --openmp flag
  bool openmp_only = false;
  std::string scene_file = "scenes/simple.txt";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--openmp") {
      openmp_only = true;
    } else {
      scene_file = arg;
    }
  }
  // END EDIT

  // Create scene
  Scene scene;

  std::cout << "Testing scene loader with: " << scene_file << "\n\n";
  // Load the scene
  scene = load_scene(scene_file);

  // Print detailed info
  // print_scene_info(scene_data);

  // =========================================================================
  // Setup camera
  // =========================================================================
  // Camera positioned at origin, looking into the scene (negative Z)
  // Camera camera(
  //     Vec3(0, 2, 5),       // Position: slightly above origin, in front of
  //     scene Vec3(0, 0, -20),     // Look at: center of the sphere arrangement
  //     60                   // Field of view: 60 degrees
  // );
  Camera camera(scene.camera.position, scene.camera.look_at, scene.camera.fov);
  // END AI EDIT

  // Framebuffer
  std::vector<Vec3> framebuffer(width * height);

  // Timing
  auto start = std::chrono::high_resolution_clock::now();

  // EDIT: Only run serial if not --openmp
  if (!openmp_only) {
    std::cout << "Rendering (Serial)...\n";

    // SERIAL VERSION
    for (int j = 0; j < height; j++) {
      // if (j % 50 == 0)
      // std::cout << "Row " << j << "/" << height << "\n";

      for (int i = 0; i < width; i++) {
        double u = double(i) / (width - 1);
        double v = double(j) / (height - 1);

        Ray ray = camera.get_ray(u, v);
        framebuffer[j * width + i] = trace_ray(ray, scene, max_depth);
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Serial time: " << diff.count() << " seconds\n";

    write_ppm("output_serial.ppm", framebuffer, width, height);
  }

// STUDENT - Add OpenMP version
// OPENMP VERSION
#ifdef _OPENMP
  std::cout << "\nRendering (OpenMP)...\n";
  // Reset the framebugger
  std::vector<Vec3> framebuffermp(width * height);
  // EDIT: declare own timing variables for OpenMP section
  auto omp_start = std::chrono::high_resolution_clock::now();
  // END EDIT

// YOUR OPENMP CODE HERE
// Hint: Use #pragma omp parallel for with appropriate scheduling
// Dynamic will likely have best performance
// Results after averaging 3 non-odd runs
// Dynamic, complex: ~0.55
// Static, complex: ~0.77
// Guided, complex: ~0.85
// Collapse idea was from claude, I had combined the loop to ij -
// then was manually calculating i and j from ij
#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      // std::cout << "ij: " << ij << ". j: " << j << "\n";

      // if (j % 50 == 0 && i == 0)
      // std::cout << "Row " << j << "/" << height << "\n";

      double u = double(i) / (width - 1);
      double v = double(j) / (height - 1);

      Ray ray = camera.get_ray(u, v);
      framebuffermp[j * width + i] = trace_ray(ray, scene, max_depth);
    }
  }

  auto omp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> omp_diff = omp_end - omp_start;
  std::cout << "OpenMP time: " << omp_diff.count() << " seconds\n";

  write_ppm("output_openmp.ppm", framebuffermp, width, height);
#endif

  return 0;
}