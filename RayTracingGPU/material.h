#pragma once
#ifndef MATERIALH
#define MATERIALH

struct hit_record; 
#include "ray.h"
//#include "hittable.h"

class material
{
public: 
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0; 
};

class lambertian : public material
{
public: 
	__device__ lambertian(const color3& a) : albedo(a) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color3& attenuation, ray& scattered, curandState* local_rand_state) const 
	{
		vec3 target = rec.p + rec.normal + random_unit_sphere(local_rand_state);
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo; 
		return true; 
	}
public: 
	color3 albedo;
};

class metal : public material
{
public: 
	__device__ metal(const color3& a, double f)
		: albedo(a) 
	{
		fuzz = f < 1 ? f : 1;
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color3& attenuation, ray& scattered, curandState* local_rand_state) const 
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		// for got what this is ?
		scattered = ray(rec.p, reflected + fuzz * random_unit_sphere(local_rand_state));
		attenuation = albedo; 
		return (dot(scattered.direction(), rec.normal) > 0.0);
	}

public: 
	color3 albedo;
	double fuzz; 
};



#endif 