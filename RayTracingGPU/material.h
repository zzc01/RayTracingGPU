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
		vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

		// Catch degenerate scatter direction 
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal; 

		scattered = ray(rec.p, scatter_direction);
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
		scattered = ray(rec.p, reflected + fuzz * random_unit_vector(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0.0);
	}

public: 
	color3 albedo;
	double fuzz; 
};

class dieletric : public material
{
public:
	__device__ dieletric(double index_of_refraction) : ir(index_of_refraction) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
									vec3& attenuation, ray& scattered,
									curandState* local_rand_state) const
	{
		attenuation = color3(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir; 
		vec3 unit_direction = unit_vector(r_in.direction()); 
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta); 

		bool cannot_refract = refraction_ratio * sin_theta > 1.0; 
		vec3 direction; 

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		scattered = ray(rec.p, direction);
		return true; 
	}
public: 
	double ir; 

private: 
	__device__ static double reflectance(double cosine, double ref_idx)
	{
		// Schlick's approx for reflectance 
		double r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0; 
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};


#endif 