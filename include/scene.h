#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "sphere.h"
#include "vec3.h"
#include "ray.h"
#include "math_constants.h"

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
    
    // Find closest sphere intersection
    bool find_intersection(const Ray& ray, double& t, int& sphere_idx) const {
        t = INFINITY_DOUBLE;
        sphere_idx = -1;
        
        // TODO: STUDENT IMPLEMENTATION
        // Loop through all spheres and find the closest intersection
        // YOUR CODE HERE
        
        return sphere_idx >= 0;
    }
    
    // Check if point is in shadow from light
    bool in_shadow(const Vec3& point, const Light& light) const {
        // TODO: STUDENT IMPLEMENTATION
        // Cast ray from point to light and check for intersections
        // YOUR CODE HERE
        
        return false;  // Placeholder
    }
    
    // Calculate color at intersection point using Phong shading
    Vec3 shade(const Vec3& point, const Vec3& normal, const Material& mat, 
               const Vec3& view_dir) const {
        Vec3 color = ambient_light * mat.color;
        
        // TODO: STUDENT IMPLEMENTATION
        // For each light:
        //   1. Check if in shadow
        //   2. Calculate diffuse component (Lambert)
        //   3. Calculate specular component (Phong)
        // YOUR CODE HERE
        
        return color;
    }
};

#endif