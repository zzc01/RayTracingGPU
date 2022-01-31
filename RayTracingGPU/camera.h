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
						double aspect )
	{
		vec3 u, v, w; 
		double theta = vfov * M_PI / 180.0; 
		double half_height = tan(theta / 2);  // this assumes distance is 1 
		double half_width = aspect * half_height; 
		origin = lookfrom; 
		w = unit_vector(lookfrom - lookat); 
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
	}
	__device__ ray get_ray(double s, double t) const
	{
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}
	point3 origin; 
	point3 lower_left_corner;
	vec3 horizontal; 
	vec3 vertical; 
};


#endif 