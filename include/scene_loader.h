#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include "scene.h"
#include "sphere.h"
#include "vec3.h"

// Camera configuration structure (separate from the Camera class in main.cpp)
struct CameraConfig {
    Vec3 position;
    Vec3 look_at;
    double fov;
    
    CameraConfig() : position(0, 0, 0), look_at(0, 0, -1), fov(60.0) {}
    CameraConfig(Vec3 pos, Vec3 target, double field_of_view) 
        : position(pos), look_at(target), fov(field_of_view) {}
};

// Result structure containing everything loaded from a scene file
struct SceneData {
    Scene scene;
    CameraConfig camera;
    bool has_camera;
    
    SceneData() : has_camera(false) {}
};

/**
 * Load a scene from a text file.
 * 
 * File format:
 *   # Comments start with #
 *   sphere x y z radius r g b metallic roughness shininess
 *   light x y z r g b intensity
 *   ambient r g b
 *   camera pos_x pos_y pos_z look_x look_y look_z fov
 * 
 * @param filename Path to the scene file
 * @return SceneData structure containing the loaded scene and camera config
 * @throws std::runtime_error if file cannot be opened or parsed
 */
inline SceneData load_scene(const std::string& filename) {
    SceneData data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open scene file: " + filename);
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) {
            continue;
        }
        line = line.substr(start);
        
        // Skip lines that start with # after trimming
        if (line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "sphere") {
            // sphere: x y z radius r g b metallic roughness shininess
            double x, y, z, radius;
            double r, g, b;
            double metallic, roughness, shininess;
            
            if (!(iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess)) {
                std::cerr << "Warning: Invalid sphere at line " << line_number 
                          << ", skipping\n";
                continue;
            }
            
            // Map scene file format to Material structure:
            // - color: RGB values
            // - reflectivity: metallic value (0.0 = diffuse, 1.0 = mirror)
            // - shininess: Phong exponent
            Material mat = {Vec3(r, g, b), metallic, shininess};
            
            data.scene.spheres.push_back(Sphere(Vec3(x, y, z), radius, mat));
        }
        else if (type == "light") {
            // light: x y z r g b intensity
            double x, y, z;
            double r, g, b;
            double intensity;
            
            if (!(iss >> x >> y >> z >> r >> g >> b >> intensity)) {
                std::cerr << "Warning: Invalid light at line " << line_number 
                          << ", skipping\n";
                continue;
            }
            
            Light light = {Vec3(x, y, z), Vec3(r, g, b), intensity};
            
            data.scene.lights.push_back(light);
        }
        else if (type == "ambient") {
            // ambient: r g b
            double r, g, b;
            
            if (!(iss >> r >> g >> b)) {
                std::cerr << "Warning: Invalid ambient at line " << line_number 
                          << ", skipping\n";
                continue;
            }
            
            data.scene.ambient_light = Vec3(r, g, b);
        }
        else if (type == "camera") {
            // camera: pos_x pos_y pos_z look_x look_y look_z fov
            double px, py, pz;
            double lx, ly, lz;
            double fov;
            
            if (!(iss >> px >> py >> pz >> lx >> ly >> lz >> fov)) {
                std::cerr << "Warning: Invalid camera at line " << line_number 
                          << ", skipping\n";
                continue;
            }
            
            data.camera = CameraConfig(Vec3(px, py, pz), Vec3(lx, ly, lz), fov);
            data.has_camera = true;
        }
        else {
            std::cerr << "Warning: Unknown type '" << type << "' at line " 
                      << line_number << ", skipping\n";
        }
    }
    
    file.close();
    
    std::cout << "Loaded scene: " << data.scene.spheres.size() << " spheres, "
              << data.scene.lights.size() << " lights\n";
    
    return data;
}

/**
 * Load a scene file by name from the scenes directory.
 * Convenience function that prepends "scenes/" to the filename.
 * 
 * @param scene_name Name of the scene (e.g., "simple", "medium", "complex")
 * @return SceneData structure containing the loaded scene and camera config
 */
inline SceneData load_scene_by_name(const std::string& scene_name) {
    return load_scene("scenes/" + scene_name + ".txt");
}

/**
 * Print scene information for debugging
 */
inline void print_scene_info(const SceneData& data) {
    std::cout << "=== Scene Information ===\n";
    std::cout << "Spheres: " << data.scene.spheres.size() << "\n";
    
    for (size_t i = 0; i < data.scene.spheres.size(); i++) {
        const Sphere& s = data.scene.spheres[i];
        std::cout << "  [" << i << "] center=(" << s.center.x << ", " 
                  << s.center.y << ", " << s.center.z << ") r=" << s.radius
                  << " color=(" << s.material.color.x << ", " 
                  << s.material.color.y << ", " << s.material.color.z << ")"
                  << " refl=" << s.material.reflectivity 
                  << " shine=" << s.material.shininess << "\n";
    }
    
    std::cout << "Lights: " << data.scene.lights.size() << "\n";
    for (size_t i = 0; i < data.scene.lights.size(); i++) {
        const Light& l = data.scene.lights[i];
        std::cout << "  [" << i << "] pos=(" << l.position.x << ", " 
                  << l.position.y << ", " << l.position.z << ")"
                  << " color=(" << l.color.x << ", " << l.color.y << ", " 
                  << l.color.z << ") intensity=" << l.intensity << "\n";
    }
    
    std::cout << "Ambient: (" << data.scene.ambient_light.x << ", " 
              << data.scene.ambient_light.y << ", " 
              << data.scene.ambient_light.z << ")\n";
    
    if (data.has_camera) {
        std::cout << "Camera: pos=(" << data.camera.position.x << ", "
                  << data.camera.position.y << ", " << data.camera.position.z 
                  << ") look_at=(" << data.camera.look_at.x << ", "
                  << data.camera.look_at.y << ", " << data.camera.look_at.z
                  << ") fov=" << data.camera.fov << "\n";
    } else {
        std::cout << "Camera: (using default)\n";
    }
    
    std::cout << "=========================\n";
}

#endif // SCENE_LOADER_H
