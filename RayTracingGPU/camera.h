#pragma once
#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
class camera
{
public: 
	__device__ camera()
	{
		lower_left_corner = point3(-2.0, -1.0, -1.0);
		horizontal = vec3(4.0, 0.0, 0.0);
		vertical = vec3(0.0, 2.0, 0.0);
		origin = point3(0.0, 0.0, 0.0); 
	}
	__device__ ray get_ray(double u, double v)
	{
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}
	point3 origin; 
	point3 lower_left_corner;
	vec3 horizontal; 
	vec3 vertical; 
};


#endif 