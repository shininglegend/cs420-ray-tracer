// demo3_shadows.cpp
// Compile: g++ -o demo3 demo3_shadows.cpp
// Usage: ./demo3

#include <iostream>
#include <cmath>
#include <vector>

struct Vec3 {
    double x, y, z;
    Vec3(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(double t) const { return Vec3(x*t, y*t, z*t); }
    double length() const { return sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const { double l = length(); return Vec3(x/l, y/l, z/l); }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
};

struct Sphere {
    Vec3 center;
    double radius;
    
    bool intersect(const Vec3& ray_origin, const Vec3& ray_dir, double& t) {
        Vec3 oc = ray_origin - center;
        double a = ray_dir.dot(ray_dir);
        double b = 2.0 * oc.dot(ray_dir);
        double c = oc.dot(oc) - radius * radius;
        double disc = b*b - 4*a*c;
        
        if (disc < 0) return false;
        
        t = (-b - sqrt(disc)) / (2*a);
        return t > 0.001;  // Small epsilon to avoid self-intersection
    }
};

int main() {
    std::cout << "=== Shadow Ray Demo ===\n\n";
    
    // Scene setup
    std::vector<Sphere> spheres = {
        {{0, 0, -5}, 1},    // Sphere 1
        {{2, 0, -3}, 0.5}   // Sphere 2 (potential occluder)
    };
    
    Vec3 light_pos(10, 10, 0);
    
    // Test points on first sphere
    Vec3 test_points[] = {
        {1, 0, -5},   // Right side of sphere 1
        {-1, 0, -5},  // Left side of sphere 1
        {0, 1, -5},   // Top of sphere 1
    };
    
    for (int i = 0; i < 3; i++) {
        Vec3 point = test_points[i];
        std::cout << "Testing point " << i+1 << ": (" 
                  << point.x << ", " << point.y << ", " << point.z << ")\n";
        
        // Cast shadow ray from point to light
        Vec3 to_light = light_pos - point;
        double light_distance = to_light.length();
        Vec3 shadow_dir = to_light.normalized();
        
        std::cout << "  Shadow ray direction: (" << shadow_dir.x << ", " 
                  << shadow_dir.y << ", " << shadow_dir.z << ")\n";
        std::cout << "  Distance to light: " << light_distance << "\n";
        
        // Check for intersection with other spheres
        bool in_shadow = false;
        for (size_t j = 0; j < spheres.size(); j++) {
            double t;
            if (spheres[j].intersect(point, shadow_dir, t)) {
                if (t < light_distance) {
                    std::cout << "  BLOCKED by sphere " << j+1 
                              << " at distance " << t << "\n";
                    in_shadow = true;
                    break;
                }
            }
        }
        
        if (!in_shadow) {
            std::cout << "  NOT in shadow - light is visible\n";
        }
        std::cout << "\n";
    }
    
    return 0;
}