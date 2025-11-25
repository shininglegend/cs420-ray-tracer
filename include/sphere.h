#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"
#include <algorithm>

struct Material
{
    Vec3 color;          // Base color
    double reflectivity; // 0.0 = diffuse, 1.0 = mirror
    double shininess;    // Phong exponent
};

class Sphere
{
public:
    Vec3 center;
    double radius;
    Material material;

    Sphere(Vec3 c, double r, Material m) : center(c), radius(r), material(m) {}

    // TODO: STUDENT IMPLEMENTATION
    // Implement ray-sphere intersection test
    // Return true if ray hits sphere, store distance in t
    // Hint: Solve quadratic equation from ray equation and sphere equation
    bool intersect(const Ray &ray, double &t) const
    {
        // 1. Calculate discriminant
        // From demo code
        Vec3 oc = ray.origin - center;
        double a = dot(ray.direction, ray.direction);
        double b = 2.0 * dot(oc, ray.direction);
        double c = dot(oc, oc) - radius * radius;

        double discriminant = b * b - 4 * a * c;

        // std::cout << "  Quadratic: " << a << "t² + " << b << "t + " << c << " = 0\n";
        // std::cout << "  Discriminant: " << discriminant;

        // 2. Check if discriminant >= 0
        if (discriminant < 0)
        {
            // std::cout << " (negative - no intersection)\n";
            return false;
        }
        // 3. Calculate t values
        if (discriminant == 0)
        {
            double t0 = -b / (2 * a);
            // Vec3 hit = ray.origin + ray.direction * t;
            // std::cout << " (zero - one intersection at t=" << t << ")\n";
            // std::cout << "  Hit point: "; hit.normalized(); std::cout << "\n";
            t = t0;
            return true;
        }
        // 4. Return smallest positive t
        double t1 = (-b - sqrt(discriminant)) / (2 * a);
        double t2 = (-b + sqrt(discriminant)) / (2 * a);
        // std::cout << " (positive - two intersections)\n";
        // std::cout << "  t1=" << t1 << ", t2=" << t2 << "\n";

        t = std::min(t1, t2); // By claude :)
        return true;
    }

    // Calculate normal at point on sphere surface
    Vec3 normal_at(const Vec3 &point) const
    {
        return (point - center).normalized();
    }
};

#endif