// math_constants.h - Mathematical Constants for Cross-Platform Compatibility
// CS420 Ray Tracer Project
// Fixes M_PI and other math constants for Visual Studio/Windows

#ifndef MATH_CONSTANTS_H
#define MATH_CONSTANTS_H

// Define M_PI if not already defined (Windows/Visual Studio compatibility)
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
    #define M_PI_2 1.57079632679489661923
#endif

#ifndef M_PI_4
    #define M_PI_4 0.78539816339744830962
#endif

// Useful constants for ray tracing
const double EPSILON = 0.001;        // Small value for avoiding self-intersection
const double INFINITY_DOUBLE = 1e20;  // Large value representing infinity
const int MAX_DEPTH = 5;             // Maximum ray recursion depth
const double RAD2DEG = 180.0 / M_PI; // Radians to degrees
const double DEG2RAD = M_PI / 180.0; // Degrees to radians

#endif // MATH_CONSTANTS_H