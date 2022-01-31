#pragma once
#ifndef HITABLEH
#define HITABLEH

//#include "ray.h"
//#include "material.h"
#include "vec3.h"
class material;

struct hit_record
{
	point3 p;
	vec3 normal; 
	double t; 
	material* mat_ptr; 
	bool front_face; 

	//// this function cannot be reached by GPU
	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable
{
public: 
	__device__ virtual bool hit(const ray & r, double t_min, double t_max, hit_record & rec) const = 0;
};

#endif 
