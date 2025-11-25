#ifndef SPHERE_H
#define SPHERE_H

#include "ray.h"
#include "vec3.h"
#include <algorithm>

struct Material {
  Vec3 color;          // Base color
  double reflectivity; // 0.0 = diffuse, 1.0 = mirror
  double shininess;    // Phong exponent
};

class Sphere {
public:
  Vec3 center;
  double radius;
  Material material;

  Sphere(Vec3 c, double r, Material m) : center(c), radius(r), material(m) {}

  // STUDENT IMPLEMENTATION
  // Implement ray-sphere intersection test
  // Return true if ray hits sphere, store distance in t
  // Hint: Solve quadratic equation from ray equation and sphere equation
  bool intersect(const Ray &ray, double &t) const {
    // 1. Calculate discriminant
    // From demo code
    Vec3 oc = ray.origin - center;
    double a = dot(ray.direction, ray.direction);
    double b = 2.0 * dot(oc, ray.direction);
    double c = dot(oc, oc) - radius * radius;

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
    if (std::max(t1, t2) < 0) {
      return false;
    }
    t = std::min(t1, t2); // By claude :)
    if (t < 0) {
      t = std::max(t1, t2);
    }
    return true;
  }

  // Calculate normal at point on sphere surface
  Vec3 normal_at(const Vec3 &point) const {
    return (point - center).normalized();
  }
};

#endif