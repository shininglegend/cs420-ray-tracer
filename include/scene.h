#ifndef SCENE_H
#define SCENE_H

#include "ray.h"
#include "ray_math_constants.h"
#include "sphere.h"
#include "vec3.h"
#include <vector>

struct Light {
  Vec3 position;
  Vec3 color;
  double intensity;
};

// Camera configuration structure (separate from the Camera class in main.cpp)
struct CameraConfig {
  Vec3 position;
  Vec3 look_at;
  double fov;

  CameraConfig() : position(0, 0, 0), look_at(0, 0, -1), fov(60.0) {}
  CameraConfig(Vec3 pos, Vec3 target, double field_of_view)
      : position(pos), look_at(target), fov(field_of_view) {}
};

class Scene {
public:
  std::vector<Sphere> spheres;
  std::vector<Light> lights;
  Vec3 ambient_light;
  CameraConfig camera;
  bool has_camera;

  // Scene() : ambient_light(0.1, 0.1, 0.1) {}

  // TODO: Should this be read from somewhere?
  double k_specular = 0.5;

  // Find closest sphere intersection
  bool find_intersection(const Ray &ray, double &t, int &sphere_idx) const {
    t = INFINITY_DOUBLE;
    sphere_idx = -1;

    // STUDENT IMPLEMENTATION (1)
    // Loop through all spheres and find the closest intersection
    // YOUR CODE HERE
    for (int curr_sphere_idx = 0; curr_sphere_idx < int(spheres.size());
         curr_sphere_idx++) {
      double temp_t = 0;
      // If an intersection, and smallest t, save it.
      if (spheres[curr_sphere_idx].intersect(ray, temp_t)) {
        if (temp_t < t) {
          sphere_idx = curr_sphere_idx;
          t = temp_t;
        }
      }
    }

    return sphere_idx >= 0;
  }

  // Check if point is in shadow from light
  // Prof. Murphy says this is fine.
  bool in_shadow(const Vec3 &point, const Light &light) const {
    // STUDENT IMPLEMENTATION (2)
    // Cast ray from point to light and check for intersections

    // BEGIN AI EDIT: Implemented shadow ray testing
    Vec3 to_light = light.position - point;
    double light_distance = to_light.length();
    Vec3 light_dir = to_light.normalized();

    // BEGIN AI EDIT: Offset shadow ray origin to avoid self-intersection (fixes
    // speckling)
    Ray shadow_ray(point + light_dir * EPSILON, light_dir);
    // END AI EDIT

    double t;
    int sphere_idx;
    if (find_intersection(shadow_ray, t, sphere_idx)) {
      return t < light_distance;
    }
    // END AI EDIT
    return false;
  }

  // Calculate color at intersection point using Phong shading
  Vec3 shade(const Vec3 &point, const Vec3 &normal, const Material &mat,
             const Vec3 &view_dir) const {
    // Start with ambient lighting
    Vec3 color = ambient_light * mat.color;

    // STUDENT IMPLEMENTATION (Usually from populi file.)
    // For each light:
    for (int light_idx = 0; light_idx < int(lights.size()); light_idx++) {
      // 1. Check if in shadow
      if (in_shadow(point, lights[light_idx])) {
        // std::cout << "Skipping...";
        continue;
      }

      // 2. Calculate diffuse component (Lambert)
      Vec3 light_dir = (lights[light_idx].position - point).normalized();
      double n_dot_l = std::max(0.0, dot(normal, light_dir));
      // BEGIN AI EDIT: Fix diffuse - use (1 - reflectivity) not reflectivity
      Vec3 diffuse = mat.color * (1.0 - mat.reflectivity) * n_dot_l;
      // END AI EDIT

      // 3. Calculate specular component (Phong)
      Vec3 reflect_dir = reflect(light_dir * -1, normal); // old
      double r_dot_v = std::max(0.0, dot(reflect_dir, view_dir));
      double spec_factor = pow(r_dot_v, mat.shininess);
      Vec3 specular = lights[light_idx].color * k_specular * spec_factor;

      // Add it all together
      color = specular + diffuse + color;
    }

    return color;
  }
};

#endif