#include "ray.h"
#include "scene.h"
#include "sphere.h"
#include "vec3.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <math_constants.h>
#include "scene_loader.h"

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
    // BEGIN AI EDIT: Offset reflection ray origin to avoid self-intersection (fixes speckling)
    Ray reflected_ray(hit + norm * EPSILON, reflected_dir);
    // END AI EDIT
    Vec3 reflected_color = trace_ray(reflected_ray, scene, depth - 1);

    double refl = scene.spheres[sphere_idx].material.reflectivity;
    shade = shade * (1.0 - refl) + reflected_color * refl;
    // END AI EDIT
  }

  return shade;
}

// Adjust for gamma 2
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
      // Apply a linear to gamma transform for gamma 2
      r = linear_to_gamma(r);
      g = linear_to_gamma(g);
      b = linear_to_gamma(b);
      r = int(255.99 * std::min(1.0, r));
      g = int(255.99 * std::min(1.0, g));
      b = int(255.99 * std::min(1.0, b));
      file << r << " " << g << " " << b << "\n";
    }
  }
}
void create_test_scene(Scene& scene) {
    // =========================================================================
    // SPHERES
    // =========================================================================
    
    // Ground "plane" - a very large sphere beneath the scene
    // Center is far below (y = -102) so only the top surface is visible
    scene.spheres.push_back(Sphere(
        Vec3(0, -102, -20),           // Center: far below the scene
        100,                          // Radius: very large
        Material{Vec3(0.5, 0.5, 0.5), 0.0, 10}  // Gray, non-reflective
    ));
    
    // Large red sphere (left) - diffuse material
    scene.spheres.push_back(Sphere(
        Vec3(-4, 0, -20),             // Left of center
        2.5,                          // Medium size
        Material{Vec3(0.9, 0.2, 0.2), 0.1, 30}  // Red, slightly reflective
    ));
    
    // Large green sphere (center) - semi-reflective
    scene.spheres.push_back(Sphere(
        Vec3(0, 0, -20),              // Center
        2.5,                          // Medium size
        Material{Vec3(0.2, 0.8, 0.2), 0.3, 50}  // Green, moderately reflective
    ));
    
    // Large blue sphere (right) - more reflective
    scene.spheres.push_back(Sphere(
        Vec3(4, 0, -20),              // Right of center
        2.5,                          // Medium size
        Material{Vec3(0.2, 0.2, 0.9), 0.5, 80}  // Blue, fairly reflective
    ));
    
    // Small white sphere (front) - highly reflective (mirror-like)
    scene.spheres.push_back(Sphere(
        Vec3(0, -1, -12),             // In front, slightly lower
        1.0,                          // Small
        Material{Vec3(0.9, 0.9, 0.9), 0.9, 200} // White, very reflective (mirror)
    ));
    
    // Small yellow sphere (back left)
    scene.spheres.push_back(Sphere(
        Vec3(-2, 1.5, -25),           // Back left, elevated
        1.5,                          // Small-medium
        Material{Vec3(0.9, 0.9, 0.2), 0.2, 40}  // Yellow
    ));
    
    // Small cyan sphere (back right)
    scene.spheres.push_back(Sphere(
        Vec3(3, 2, -28),              // Back right, elevated
        1.5,                          // Small-medium
        Material{Vec3(0.2, 0.9, 0.9), 0.2, 40}  // Cyan
    ));
    
    // =========================================================================
    // LIGHTS
    // =========================================================================
    
    // Main key light (upper right, warm white)
    Light key_light;
    key_light.position = Vec3(10, 10, -5);
    key_light.color = Vec3(1.0, 0.95, 0.9);   // Slightly warm white
    key_light.intensity = 0.8;
    scene.lights.push_back(key_light);
    
    // Fill light (upper left, cool blue)
    // Provides softer illumination on the shadow side
    Light fill_light;
    fill_light.position = Vec3(-10, 8, -5);
    fill_light.color = Vec3(0.8, 0.9, 1.0);   // Slightly cool/blue
    fill_light.intensity = 0.4;
    scene.lights.push_back(fill_light);
    
    // Rim light (behind, adds edge definition)
    Light rim_light;
    rim_light.position = Vec3(0, 5, -35);
    rim_light.color = Vec3(1.0, 1.0, 1.0);
    rim_light.intensity = 0.3;
    scene.lights.push_back(rim_light);
    
    // =========================================================================
    // AMBIENT LIGHT
    // =========================================================================
    // Low ambient prevents completely black shadows
    scene.ambient_light = Vec3(0.1, 0.1, 0.12);  // Slightly blue ambient
}



int main(int argc, char *argv[]) {
  // Image settings
  const int width = 640;
  const int height = 480;
  const int max_depth = 10;

  // Create scene
  Scene scene;
    SceneData scene_data;
    std::string scene_file = "scenes/simple.txt";
    
    // Allow command-line scene selection
    if (argc > 1) {
        scene_file = argv[1];
    }
    
    std::cout << "Testing scene loader with: " << scene_file << "\n\n";
    // Load the scene
    scene_data = load_scene(scene_file);
    scene = scene_data.scene;
    
    // Print detailed info
    print_scene_info(scene_data);
    
    // =========================================================================
    // Setup camera
    // =========================================================================
    // Camera positioned at origin, looking into the scene (negative Z)
    // Camera camera(
    //     Vec3(0, 2, 5),       // Position: slightly above origin, in front of scene
    //     Vec3(0, 0, -20),     // Look at: center of the sphere arrangement
    //     60                   // Field of view: 60 degrees
    // );
    Camera camera(
        scene_data.camera.position,
        scene_data.camera.look_at,
        scene_data.camera.fov
    );
  // END AI EDIT

  // Framebuffer
  std::vector<Vec3> framebuffer(width * height);
  std::cout << "Rendering (Serial)...\n";

  // Timing
  auto start = std::chrono::high_resolution_clock::now();

  // SERIAL VERSION
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
  // Reset the framebugger
  std::vector<Vec3> framebuffermp(width * height);
  start = std::chrono::high_resolution_clock::now();

// YOUR OPENMP CODE HERE
// Hint: Use #pragma omp parallel for with appropriate scheduling
// Dynamic will likely have best performance
// Results after averaging 3 non-odd runs
// Dynamic, complex: ~0.55
// Static, complex: ~0.77
// Guided, complex: ~0.85
#pragma omp parallel for schedule(dynamic)
  for (int ij = 0; ij < (height * width); ij++) {
    int j = floor(double(ij) / width);
    int i = ij % width;
    // std::cout << "ij: " << ij << ". j: " << j << "\n";

    if (j % 50 == 0 && i == 0)
      std::cout << "Row " << j << "/" << height << "\n";

    double u = double(i) / (width - 1);
    double v = double(j) / (height - 1);

    Ray ray = camera.get_ray(u, v);
    framebuffermp[j * width + i] = trace_ray(ray, scene, max_depth);
  }

  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "OpenMP time: " << diff.count() << " seconds\n";

  write_ppm("output_openmp.ppm", framebuffermp, width, height);
#endif

  return 0;
}