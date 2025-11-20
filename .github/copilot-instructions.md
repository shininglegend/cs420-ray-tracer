# CS420 Ray Tracer - AI Agent Instructions

## Project Overview
Academic ray tracer implementing progressive parallelization: Serial → OpenMP → CUDA → Hybrid. This is a **student project** with intentional TODO markers - completeness varies by implementation phase.

### IMPORTANT
- This project is for a systems and parallel processing class. 
- Whenever you edit code, you MUST add a comment at the beginning and end of your edit showing your edits (minimal comment is fine.) 

## Architecture & Data Flow

### Core Ray Tracing Pipeline
1. **Camera** (`include/camera.h`) generates rays for each pixel (u,v coordinates)
2. **Scene::find_intersection()** finds closest sphere hit across all geometry
3. **Scene::shade()** applies Phong lighting (ambient + diffuse + specular)
4. **trace_ray()** recursively handles reflections (depth-limited)
5. Output written as PPM format to `output_*.ppm`

### Key Components
- **Vec3** (`include/vec3.h`): Foundation 3D vector with dot/cross/normalize ops
- **Ray** (`include/ray.h`): Origin + direction representation
- **Sphere** (`include/sphere.h`): Quadratic equation solver for ray-sphere intersection
- **Material**: Phong properties (albedo, metallic, roughness, shininess)
- **Scene**: Manages spheres/lights, implements shadow rays and shading
- **Light**: Point lights with position/color/intensity

## Critical Implementation Patterns

### 1. Ray-Sphere Intersection (Quadratic Equation)
```cpp
// From include/sphere.h - solve: ||ray.at(t) - center||² = radius²
Vec3 oc = ray.origin - center;
double a = dot(ray.direction, ray.direction);
double b = 2.0 * dot(oc, ray.direction);
double c = dot(oc, oc) - radius * radius;
double discriminant = b*b - 4*a*c;
// Return smallest positive t value
```

### 2. Phong Shading Model (Scene::shade)
```cpp
Vec3 color = ambient_light * mat.color;  // Ambient
// For each light (if not in_shadow):
//   diffuse += light_color * max(0, N·L)
//   specular += light_color * pow(max(0, R·V), shininess)
```

### 3. Shadow Ray Testing
Cast ray from hit_point toward light - if ANY intersection before reaching light, point is in shadow.

### 4. OpenMP Parallelization Strategy
```cpp
#pragma omp parallel for schedule(dynamic) collapse(2)
for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
        // Ray trace pixel - thread-safe (no shared writes)
```
**Critical**: Use `schedule(dynamic)` - pixels near scene center have more sphere tests (variable cost). See `populi-files/demo4_openmp_scheduling.cpp` for proof.

## Build System & Workflows

### Primary Commands
```bash
make serial   # Week 1: Single-threaded baseline
make openmp   # Week 1: Multi-threaded with -fopenmp
make cuda     # Week 2: GPU version from main_gpu.cu
make hybrid   # Week 3: Combined OpenMP + CUDA
make benchmark  # Compare serial vs openmp performance
```

### Scene Files Format
```
# scenes/*.txt - Plain text scene description
sphere x y z radius r g b metallic roughness shininess
light x y z r g b intensity
ambient r g b
camera x y z look_x look_y look_z fov
```

### Testing Scripts
- `scripts/test.sh`: Quick validation run
- `scripts/benchmark.sh`: Multi-iteration performance comparison with CSV output

## UNIX-Specific Conventions

### Platform Considerations
- **Math constants**: `M_PI` available directly on UNIX (no `_USE_MATH_DEFINES` needed)
- **OpenMP guards**: Always wrap OpenMP code with `#ifdef _OPENMP` for clean compilation without `-fopenmp`
- **File structure**: `include/` for headers, `src/` for implementations (makefile uses `-I$(INCDIR)`)

### Compilation Flags
- C++11 standard (`-std=c++11`)
- High optimization (`-O3`)

### Demo Files Reference
`populi-files/` contains working examples:
- `demo1_intersection.cpp`: Ray-sphere math walkthrough
- `demo2_phong.cpp`: Lighting model reference
- `demo3_shadows.cpp`: Shadow ray implementation
- `demo4_openmp_scheduling.cpp`: Proof that dynamic scheduling outperforms static

## Code Generation Guidelines

### When Adding/Fixing Features
1. **Check for existing placeholders**: Look for `// YOUR CODE HERE` or loop conditions like `i < 0`
2. **Maintain the educational structure**: Keep TODO comments, don't over-engineer, ensure student understanding
3. **Test incrementally**: Ask the student to build with `make serial`, generally test with single sphere before complex scenes
4. **Preserve debug output**: Student code may have `std::cout` statements in sphere.h - keep unless explicitly told to remove
5. **Use Vec3 operations**: Always use `dot()`, `cross()`, `normalized()` methods rather than raw math

### GPU/CUDA Specifics (Week 2+)
- Use `float3` and `float3_ops` helper struct (defined in `main_gpu.cu`)
- Mark device functions with `__device__`
- Use `CUDA_CHECK()` macro for error handling
- Allocate scene data on device with `cudaMalloc`/`cudaMemcpy`

### Always check and note common pitfalls

## Expected Outputs
- `output_serial.ppm`: Single-threaded render
- `output_openmp.ppm`: Parallel render (should match serial visually)
- Performance target: OpenMP ~4-8x speedup on 8-core systems with `dynamic` scheduling
