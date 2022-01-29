#pragma once
#ifndef SPHEREH 
#define SPHEREH 

#include "hittable.h"

class sphere : public hittable
{
public: 
	__device__ sphere() {}
	__device__ sphere(point3 cen, double r) : center(cen), radius(r) {}
	__device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

	point3 center;
	double radius; 
};

__device__ bool sphere::hit(const ray& r, double tmin, double tmax, hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	double a = r.direction().squared_length();
	double half_b = dot(oc, r.direction());
	double c = oc.squared_length() - radius * radius;
	double discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false; 
	// 
	double root = (-half_b - sqrt(discriminant)) / a;
	if (root < tmax && root > tmin)
	{
		rec.t = root; 
		rec.p = r.at(root);
		rec.normal = (rec.p - center) / radius; 
		return true; 
	}
	//
	root = (-half_b + sqrt(discriminant)) / a;
	if (root < tmax && root > tmin)
	{
		rec.t = root;
		rec.p = r.at(root);
		rec.normal = (rec.p - center) / radius;
		return true;
	}
}

#endif // !SPHEREH 
