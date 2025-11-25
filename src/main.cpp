#include "ray.h"
#include "scene.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

class Camera {
public:
  Vec3 position;
  Vec3 forward, right, up;
  double fov;

  Camera(Vec3 pos, Vec3 look_at, double field_of_view)
      : position(pos), fov(field_of_view) {
    forward = (look_at - position).normalized();
    right = cross(forward, Vec3(0, 1, 0)).normalized();
    up = cross(right, forward).normalized();
  }

  Ray get_ray(double u, double v) const {
    double aspect = 1.0;
    double scale = tan(fov * 0.5 * M_PI / 180.0);

    Vec3 direction = forward + right * ((u - 0.5) * scale * aspect) +
                     up * ((v - 0.5) * scale);

    return Ray(position, direction.normalized());
  }
};

// Trace a single ray through the scene
Vec3 trace_ray(const Ray &ray, const Scene &scene, int depth) {
  if (depth <= 0)
    return Vec3(0, 0, 0);

  // TODO: STUDENT IMPLEMENTATION (4)
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
  Vec3 shade = scene.shade(hit, norm, scene.spheres[sphere_idx].material, Vec3());

  // 3. If material is reflective, recursively trace reflection ray
  if (scene.spheres[sphere_idx].material.reflectivity > 0) {
    // TODO: Fill this out
  }

  return shade;
}

// Write image to PPM file
void write_ppm(const std::string &filename,
               const std::vector<Vec3> &framebuffer, int width, int height) {
  std::ofstream file(filename);
  file << "P3\n" << width << " " << height << "\n255\n";

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      Vec3 color = framebuffer[j * width + i];
      int r = int(255.99 * std::min(1.0, color.x));
      int g = int(255.99 * std::min(1.0, color.y));
      int b = int(255.99 * std::min(1.0, color.z));
      file << r << " " << g << " " << b << "\n";
    }
  }
}

int main(int argc, char *argv[]) {
  // Image settings
  const int width = 640;
  const int height = 480;
  const int max_depth = 3;

  // Create scene
  Scene scene;

  // TODO: STUDENT - Add spheres to scene
  // Example:
  scene.spheres.push_back(
      Sphere(Vec3(0, 0, -20), 2, Material{Vec3(1, 0, 0), 0.5, 50}));

  // TODO: STUDENT - Add lights to scene
  // Example:
  scene.lights.push_back(Light{Vec3(10, 10, -10), Vec3(1, 1, 1), 1.0});

  // Setup camera
  Camera camera(Vec3(0, 0, 0), Vec3(0, 0, -1), 60);

  // Framebuffer
  std::vector<Vec3> framebuffer(width * height);

  // Timing
  auto start = std::chrono::high_resolution_clock::now();

  // SERIAL VERSION
  std::cout << "Rendering (Serial)...\n";
  for (int j = 0; j < height; j++) {
    if (j % 50 == 0)
      std::cout << "Row " << j << "/" << height << "\n";

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

// TODO: STUDENT - Add OpenMP version
// OPENMP VERSION
#ifdef _OPENMP
  std::cout << "\nRendering (OpenMP)...\n";
  start = std::chrono::high_resolution_clock::now();

  // YOUR OPENMP CODE HERE
  // Hint: Use #pragma omp parallel for with appropriate scheduling
  // Dynamic will likely have best performance

  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "OpenMP time: " << diff.count() << " seconds\n";

  write_ppm("output_openmp.ppm", framebuffer, width, height);
#endif

  return 0;
}