#pragma once
#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
#define M_PI 3.14159265359

class camera
{
public: 
	__device__ camera(  point3 lookfrom,
						point3 lookat, 
						vec3 vup, 
						double vfov, 
						double aspect,
						double aperatur, 
						double focus_dist)
	{
		double theta = vfov * M_PI / 180.0; 
		double half_height = tan(theta / 2) * focus_dist;  // this assumes distance is 1 
		double half_width = aspect * half_height; 
		origin = lookfrom; 
		w = unit_vector(lookfrom - lookat); 
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w * focus_dist;
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
		lens_radius = aperatur / 2; 
	}
	__device__ ray get_ray(double s, double t, curandState* local_rand_state) const
	{
		vec3 rd = lens_radius * random_in_unit_disk(local_rand_state); 
		vec3 offset = u * rd.x() + v * rd.y(); 
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}
	point3 origin; 
	point3 lower_left_corner;
	vec3 horizontal; 
	vec3 vertical; 
	vec3 u, v, w;
	double lens_radius;
};


#endif 