#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "scene.h"
#include "sphere.h"
#include "vec3.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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
inline Scene load_scene(const std::string &filename) {
  Scene scene;
  std::ifstream file(filename);
  scene.has_camera = false;

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

      if (!(iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >>
            roughness >> shininess)) {
        std::cerr << "Warning: Invalid sphere at line " << line_number
                  << ", skipping\n";
        continue;
      }

      // Map scene file format to Material structure:
      // - color: RGB values
      // - reflectivity: metallic value (0.0 = diffuse, 1.0 = mirror)
      // - shininess: Phong exponent
      Material mat = {Vec3(r, g, b), metallic, shininess};

      scene.spheres.push_back(Sphere(Vec3(x, y, z), radius, mat));
    } else if (type == "light") {
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

      scene.lights.push_back(light);
    } else if (type == "ambient") {
      // ambient: r g b
      double r, g, b;

      if (!(iss >> r >> g >> b)) {
        std::cerr << "Warning: Invalid ambient at line " << line_number
                  << ", skipping\n";
        continue;
      }

      scene.ambient_light = Vec3(r, g, b);
    } else if (type == "camera") {
      // camera: pos_x pos_y pos_z look_x look_y look_z fov
      double px, py, pz;
      double lx, ly, lz;
      double fov;

      if (!(iss >> px >> py >> pz >> lx >> ly >> lz >> fov)) {
        std::cerr << "Warning: Invalid camera at line " << line_number
                  << ", skipping\n";
        continue;
      }

      scene.camera = CameraConfig(Vec3(px, py, pz), Vec3(lx, ly, lz), fov);
      scene.has_camera = true;
    } else {
      std::cerr << "Warning: Unknown type '" << type << "' at line "
                << line_number << ", skipping\n";
    }
  }

  file.close();

  std::cout << "Loaded scene: " << scene.spheres.size() << " spheres, "
            << scene.lights.size() << " lights\n";

  return scene;
}

/**
 * Load a scene file by name from the scenes directory.
 * Convenience function that prepends "scenes/" to the filename.
 *
 * @param scene_name Name of the scene (e.g., "simple", "medium", "complex")
 * @return Scene structure containing the loaded scene
 */
inline Scene load_scene_by_name(const std::string &scene_name) {
  return load_scene("scenes/" + scene_name + ".txt");
}

/**
 * Print scene information for debugging
 */
inline void print_scene_info(const Scene &scene) {
  std::cout << "=== Scene Information ===\n";
  std::cout << "Spheres: " << scene.spheres.size() << "\n";

  for (size_t i = 0; i < scene.spheres.size(); i++) {
    const Sphere &s = scene.spheres[i];
    std::cout << "  [" << i << "] center=(" << s.center.x << ", " << s.center.y
              << ", " << s.center.z << ") r=" << s.radius << " color=("
              << s.material.color.x << ", " << s.material.color.y << ", "
              << s.material.color.z << ")"
              << " refl=" << s.material.reflectivity
              << " shine=" << s.material.shininess << "\n";
  }

  std::cout << "Lights: " << scene.lights.size() << "\n";
  for (size_t i = 0; i < scene.lights.size(); i++) {
    const Light &l = scene.lights[i];
    std::cout << "  [" << i << "] pos=(" << l.position.x << ", " << l.position.y
              << ", " << l.position.z << ")"
              << " color=(" << l.color.x << ", " << l.color.y << ", "
              << l.color.z << ") intensity=" << l.intensity << "\n";
  }

  std::cout << "Ambient: (" << scene.ambient_light.x << ", "
            << scene.ambient_light.y << ", " << scene.ambient_light.z << ")\n";

  if (scene.has_camera) {
    std::cout << "Camera: pos=(" << scene.camera.position.x << ", "
              << scene.camera.position.y << ", " << scene.camera.position.z
              << ") look_at=(" << scene.camera.look_at.x << ", "
              << scene.camera.look_at.y << ", " << scene.camera.look_at.z
              << ") fov=" << scene.camera.fov << "\n";
  } else {
    std::cout << "Camera: (using default)\n";
  }

  std::cout << "=========================\n";
}

#endif // SCENE_LOADER_H
