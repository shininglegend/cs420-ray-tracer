#include "ray.h"
#include "vec3.h"

class Camera {
public:
  Vec3 position;
  Vec3 forward, right, up;
  double fov;

  Camera(Vec3 pos, Vec3 look_at, double field_of_view)
      : position(pos), fov(field_of_view) {
    forward = (look_at - position).normalized();
    right = cross(forward, Vec3(0, 1, 0)).normalized();
    up = cross(right, forward).normalized();
  }

  Ray get_ray(double u, double v) const {
    double aspect = 1.0;
    double scale = tan(fov * 0.5 * M_PI / 180.0);

    Vec3 direction = forward + right * ((u - 0.5) * scale * aspect) +
                     up * ((v - 0.5) * scale);

    return Ray(position, direction.normalized());
  }
};