// demo2_phong.cpp
// Compile: g++ -o demo2 demo2_phong.cpp
// Usage: ./demo2

#include <iostream>
#include <cmath>
#include <algorithm>

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

Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2 * v.dot(n));
}

int main() {
    std::cout << "=== Phong Shading Demo ===\n\n";
    
    // Surface point and normal
    Vec3 point(0, 0, 0);
    Vec3 normal(0, 1, 0);  // Pointing up
    
    // Material properties
    Vec3 material_color(1, 0, 0);  // Red
    double k_ambient = 0.1;
    double k_diffuse = 0.7;
    double k_specular = 0.5;
    double shininess = 32;
    
    // Light
    Vec3 light_pos(5, 5, 5);
    Vec3 light_color(1, 1, 1);
    
    // Camera
    Vec3 camera_pos(0, 5, 5);
    
    std::cout << "Setup:\n";
    std::cout << "  Surface normal: (0, 1, 0) [pointing up]\n";
    std::cout << "  Material: Red with shininess=" << shininess << "\n";
    std::cout << "  Light at: (5, 5, 5)\n";
    std::cout << "  Camera at: (0, 5, 5)\n\n";
    
    // Calculate lighting step by step
    std::cout << "Calculations:\n";
    
    // 1. Ambient
    Vec3 ambient = material_color * k_ambient;
    std::cout << "1. Ambient = material * k_a = (" 
              << ambient.x << ", " << ambient.y << ", " << ambient.z << ")\n";
    
    // 2. Diffuse
    Vec3 light_dir = (light_pos - point).normalized();
    double n_dot_l = std::max(0.0, normal.dot(light_dir));
    Vec3 diffuse = material_color * k_diffuse * n_dot_l;
    
    std::cout << "2. Light direction: (" << light_dir.x << ", " 
              << light_dir.y << ", " << light_dir.z << ")\n";
    std::cout << "   N·L = " << n_dot_l << "\n";
    std::cout << "   Diffuse = material * k_d * (N·L) = (" 
              << diffuse.x << ", " << diffuse.y << ", " << diffuse.z << ")\n";
    
    // 3. Specular
    Vec3 view_dir = (camera_pos - point).normalized();
    Vec3 reflect_dir = reflect(light_dir * -1, normal);
    double r_dot_v = std::max(0.0, reflect_dir.dot(view_dir));
    double spec_factor = pow(r_dot_v, shininess);
    Vec3 specular = light_color * k_specular * spec_factor;
    
    std::cout << "3. View direction: (" << view_dir.x << ", " 
              << view_dir.y << ", " << view_dir.z << ")\n";
    std::cout << "   Reflection direction: (" << reflect_dir.x << ", " 
              << reflect_dir.y << ", " << reflect_dir.z << ")\n";
    std::cout << "   R·V = " << r_dot_v << "\n";
    std::cout << "   (R·V)^" << shininess << " = " << spec_factor << "\n";
    std::cout << "   Specular = light * k_s * (R·V)^n = (" 
              << specular.x << ", " << specular.y << ", " << specular.z << ")\n";
    
    // Total
    Vec3 total = ambient + diffuse + specular;
    std::cout << "\nFinal Color = Ambient + Diffuse + Specular\n";
    std::cout << "            = (" << total.x << ", " << total.y << ", " 
              << total.z << ")\n";
    
    // Convert to RGB
    int r = std::min(255, (int)(total.x * 255));
    int g = std::min(255, (int)(total.y * 255));
    int b = std::min(255, (int)(total.z * 255));
    std::cout << "RGB values: (" << r << ", " << g << ", " << b << ")\n";
    
    return 0;
}