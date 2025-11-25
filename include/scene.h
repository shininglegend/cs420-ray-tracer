#ifndef SCENE_H
#define SCENE_H

#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <vector>

struct Light {
  Vec3 position;
  Vec3 color;
  double intensity;
};

class Scene {
public:
  std::vector<Sphere> spheres;
  std::vector<Light> lights;
  Vec3 ambient_light;

  Scene() : ambient_light(0.1, 0.1, 0.1) {}

  //   Vec3 reflect(const Vecx3 &v, const Vec3 &n) { return v - n * (2 *
  //   v.dot(n)); }

  // Find closest sphere intersection
  bool find_intersection(const Ray &ray, double &t, int &sphere_idx) const {
    t = INFINITY;
    sphere_idx = -1;

    // TODO: STUDENT IMPLEMENTATION (1)
    // Loop through all spheres and find the closest intersection
    // YOUR CODE HERE
    for (int curr_sphere_idx = 0; curr_sphere_idx < spheres.size();
         curr_sphere_idx++) {
      double temp_t = 0;
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
  bool in_shadow(const Vec3 &point, const Light &light) const {
    // TODO: STUDENT IMPLEMENTATION (2)
    // Cast ray from point to light and check for intersections
    // YOUR CODE HERE

    // BEGIN AI EDIT: Implemented shadow ray testing
    Vec3 light_dir = (light.position - point).normalized();

    // Offset this to avoid self-intersection
    Ray shadow_ray(point + light_dir * 0.001, light_dir);
    double light_distance = (light.position - point).length();

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
    Vec3 color = ambient_light * mat.color;

    // TODO: STUDENT IMPLEMENTATION (3)
    // For each light:
    //   1. Check if in shadow
    //   2. Calculate diffuse component (Lambert)
    //   3. Calculate specular component (Phong)
    // YOUR CODE HERE
    for (int light_idx = 0; light_idx < lights.size(); light_idx++) {
      if (in_shadow(point, lights[light_idx])) {
        continue;
      }
      //   // 2. Diffuse (Lambert)
      //   Vec3 light_dir = (lights[light_idx].position - point).normalized();
      //   double n_dot_l = std::max(0.0, dot(normal, light_dir));
      //   Vec3 diffuse = mat.color * mat.reflectivity * n_dot_l;

      //   std::cout << "2. Light direction: (" << light_dir.x << ", " <<
      //   light_dir.y
      //             << ", " << light_dir.z << ")\n";
      //   std::cout << "   N·L = " << n_dot_l << "\n";
      //   std::cout << "   Diffuse = material * k_d * (N·L) = (" << diffuse.x
      //             << ", " << diffuse.y << ", " << diffuse.z << ")\n";

      //   // 3. Phong / Specular
      //   Vec3 reflect_dir = reflect(light_dir * -1, normal);
      //   double r_dot_v = std::max(0.0, dot(reflect_dir, view_dir));
      //   double spec_factor = pow(r_dot_v, mat.shininess);
      //   Vec3 specular = lights[light_idx].color * k_specular * spec_factor;

      //   std::cout << "3. View direction: (" << view_dir.x << ", " <<
      //   view_dir.y
      //             << ", " << view_dir.z << ")\n";
      //   std::cout << "   Reflection direction: (" << reflect_dir.x << ", "
      //             << reflect_dir.y << ", " << reflect_dir.z << ")\n";
      //   std::cout << "   R·V = " << r_dot_v << "\n";
      //   std::cout << "   (R·V)^" << shininess << " = " << spec_factor <<
      //   "\n"; std::cout << "   Specular = light * k_s * (R·V)^n = (" <<
      //   specular.x
      //             << ", " << specular.y << ", " << specular.z << ")\n";
    }

    return color;
  }
};

#endif