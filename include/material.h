// material.h - Material Properties and Shading (Fixed Version)
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE
// Fixed for Visual Studio compatibility

#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"
#include "ray.h"
#include <algorithm>  // for std::max

// Material types
enum MaterialType {
    DIFFUSE,     // Lambertian diffuse
    METAL,       // Reflective metal
    DIELECTRIC   // Glass/transparent (optional extra credit)
};

class Material {
public:
    Vec3 albedo;           // Base color/reflectance
    double metallic;       // 0 = diffuse, 1 = perfect mirror
    double roughness;      // Surface roughness (affects specular)
    double shininess;      // Phong exponent
    MaterialType type;
    
    // Default constructor - red diffuse material
    Material() 
        : albedo(0.5, 0.5, 0.5), metallic(0.0), roughness(0.5), 
          shininess(32.0), type(DIFFUSE) {}
    
    // Constructor for specific material
    Material(const Vec3& color, double met, double rough, double shine, MaterialType t = DIFFUSE)
        : albedo(color), metallic(met), roughness(rough), 
          shininess(shine), type(t) {}
    
    // Simple constructors for common materials
    static Material diffuse(const Vec3& color) {
        return Material(color, 0.0, 1.0, 10.0, DIFFUSE);
    }
    
    static Material metal(const Vec3& color, double roughness = 0.3) {
        return Material(color, 1.0, roughness, 100.0, METAL);
    }
    
    static Material shiny(const Vec3& color, double shininess = 50.0) {
        return Material(color, 0.5, 0.3, shininess, DIFFUSE);
    }
    
    // =========================================================
    // TODO: STUDENT IMPLEMENTATION - Phong Shading Model
    // =========================================================
    // Implement the Phong illumination model with the formula:
    // 
    // I_total = I_ambient + I_diffuse + I_specular
    // 
    // Where:
    // - I_ambient = k_a * ambient_light_color
    // - I_diffuse = k_d * light_color * max(0, N·L)
    // - I_specular = k_s * light_color * max(0, R·V)^n
    // 
    // Parameters:
    // - hit_point: Point on surface being shaded
    // - normal: Surface normal at hit point
    // - view_dir: Direction from hit point to camera
    // - light_pos: Position of light source
    // - light_color: Color/intensity of light
    // - ambient: Ambient light in scene
    // =========================================================
    
    Vec3 shade(const Vec3& hit_point, const Vec3& normal, const Vec3& view_dir,
               const Vec3& light_pos, const Vec3& light_color, 
               const Vec3& ambient) const {
        
        // TODO: STUDENT CODE HERE
        // Steps:
        // 1. Calculate ambient component: ambient * albedo * (1-metallic)
        // 2. Calculate light direction: (light_pos - hit_point).normalized()
        // 3. Calculate diffuse component: albedo * light_color * max(0, N·L)
        // 4. Calculate reflection direction: reflect(-light_dir, normal)
        // 5. Calculate specular component: light_color * max(0, R·V)^shininess
        // 6. Combine components based on material properties
        //
        // Example structure:
        // Vec3 color = ambient * albedo * (1.0 - metallic);  // Ambient
        // 
        // Vec3 light_dir = (light_pos - hit_point).normalized();
        // double n_dot_l = std::max(0.0, dot(normal, light_dir));
        // color = color + albedo * light_color * n_dot_l * (1.0 - metallic);  // Diffuse
        // 
        // Vec3 reflect_dir = reflect(-light_dir, normal);
        // double r_dot_v = std::max(0.0, dot(reflect_dir, view_dir));
        // double spec = pow(r_dot_v, shininess);
        // color = color + light_color * spec * metallic;  // Specular
        // 
        // return color;
        
        // PLACEHOLDER - Remove this and implement Phong shading
        return albedo;  // Just return base color for now
    }
    
    // Check if material should produce reflections
    bool is_reflective() const {
        return metallic > 0.1;
    }
    
    // Get reflection coefficient
    double get_reflectivity() const {
        return metallic;
    }
};

#endif // MATERIAL_H