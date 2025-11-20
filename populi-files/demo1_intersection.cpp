// demo1_intersection.cpp
// Compile: g++ -o demo1 demo1_intersection.cpp

// Usage: ./demo1

#include <iostream>
#include <cmath>

struct Vec3
{
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double t) const { return Vec3(x * t, y * t, z * t); }
    double dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
    void print() const { std::cout << "(" << x << ", " << y << ", " << z << ")"; }
};

// Interactive demonstration of ray-sphere intersection
int main()
{
    std::cout << "=== Ray-Sphere Intersection Demo ===\n\n";

    // Sphere at origin with radius 2
    Vec3 sphere_center(0, 0, 0);
    double radius = 1.0;

    // Test different rays
    struct TestRay
    {
        Vec3 origin, direction;
        const char *description;
    };

    TestRay test_rays[] = {
        {{5, 0, 0}, {-1, 0, 0}, "Ray along X-axis (hits)"},
        {{5, 5, 0}, {-1, 0, 0}, "High horizontal ray (misses)"},
        {{5, 1, 0}, {-1, 0, 0}, "Offset ray (grazing)"},
        {{0, 0, 5}, {0, 0, -1}, "Ray along Z-axis (hits)"}};

    for (const auto &test : test_rays)
    {
        std::cout << test.description << "\n";
        std::cout << "  Ray origin: ";
        test.origin.print();
        std::cout << "  Direction: ";
        test.direction.print();
        std::cout << "\n";

        // Calculate intersection
        Vec3 oc = test.origin - sphere_center;
        double a = test.direction.dot(test.direction);
        double b = 2.0 * oc.dot(test.direction);
        double c = oc.dot(oc) - radius * radius;

        double discriminant = b * b - 4 * a * c;

        std::cout << "  Quadratic: " << a << "t² + " << b << "t + " << c << " = 0\n";
        std::cout << "  Discriminant: " << discriminant;

        if (discriminant < 0)
        {
            std::cout << " (negative - no intersection)\n";
        }
        else if (discriminant == 0)
        {
            // Only one intersection for this case
            double t = -b / (2 * a);
            Vec3 hit = test.origin + test.direction * t;
            std::cout << " (zero - one intersection at t=" << t << ")\n";
            std::cout << "  Hit point: ";
            hit.print();
            std::cout << "\n";
        }
        else
        {
            double t1 = (-b - sqrt(discriminant)) / (2 * a);
            double t2 = (-b + sqrt(discriminant)) / (2 * a);
            std::cout << " (positive - two intersections)\n";
            std::cout << "  t1=" << t1 << ", t2=" << t2 << "\n";

            if (t1 > 0)
            {
                Vec3 hit = test.origin + test.direction * t1;
                std::cout << "  Nearest hit: ";
                hit.print();
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    return 0;
}