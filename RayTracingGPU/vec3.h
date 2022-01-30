#pragma once
#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <curand_kernel.h>

class vec3
{
public: 
	// what does it mean by having this function 
	// device callable, but how about the members
	// like member variables? or member functions? 
	__host__ __device__ vec3() {}
	__host__ __device__ vec3(double e0, double e1, double e2)
		: e{e0, e1, e2} {}
	__host__ __device__ inline double x() const { return e[0]; }
	__host__ __device__ inline double y() const { return e[1]; }
	__host__ __device__ inline double z() const { return e[2]; }
	__host__ __device__ inline double r() const { return e[0]; }
	__host__ __device__ inline double g() const { return e[1]; }
	__host__ __device__ inline double b() const { return e[2]; }


	__host__ __device__ inline const vec3& operator+() const { return *this; }
	__host__ __device__ inline const vec3& operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline double operator[](int i) const { return e[i]; }
	__host__ __device__ inline double& operator[](int i) { return e[i]; }

	// declare these here and define it later 
	__host__ __device__ inline vec3& operator+=(const vec3& v2);
	__host__ __device__ inline vec3& operator-=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const vec3& v2);
	__host__ __device__ inline vec3& operator/=(const vec3& v2);
	__host__ __device__ inline vec3& operator*=(const double t);
	__host__ __device__ inline vec3& operator/=(const double t);

	__host__ __device__ inline double squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ inline double length() const { return sqrt(squared_length()); }
	// if there any different from this and above? 
	//__host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }

	// look like a place holder 
	__host__ __device__ inline void make_unit_vector();

	double e[3];
};

// Type aliases for vec3 
using color3 = vec3;
using point3 = vec3;


// ================================ 

__host__ __device__ inline vec3& vec3::operator+=(const vec3 & v2)
{
	e[0] += v2.e[0];
	e[1] += v2.e[1];
	e[2] += v2.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v2)
{
	e[0] -= v2.e[0];
	e[1] -= v2.e[1];
	e[2] -= v2.e[2];
	return *this;
}
__host__ __device__ inline vec3& vec3::operator*=(const vec3& v2)
{
	e[0] *= v2.e[0]; 
	e[1] *= v2.e[1]; 
	e[2] *= v2.e[2]; 
	return *this;
}
__host__ __device__ inline vec3& vec3::operator/=(const vec3& v2)
{
	e[0] /= v2.e[0]; 
	e[1] /= v2.e[1]; 
	e[2] /= v2.e[2]; 
	return *this;
}
__host__ __device__ inline vec3& vec3::operator*=(const double t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const double t)
{
	return (*this) *= (1.0 / t);
}

__host__ __device__ inline void vec3::make_unit_vector()
{
	double k = 1.0 / length();
	vec3 unit_v = (*this) *= k;
}

// ================================ 

inline std::istream& operator>>(std::istream& is, vec3& v)
{
	is >> v.e[0] >> v.e[0] >> v.e[2];
	return is; 
}

inline std::ostream& operator<<(std::ostream& os, vec3& v)
{
	os << v.e[0] << " " << v.e[0] << " " << v.e[2];
	return os;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, double t)
{
	return vec3(v1.e[0] * t, v1.e[1] * t, v1.e[2] * t);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v1)
{
	return v1 * t;
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, double t)
{
	return vec3(v1.e[0] / t, v1.e[1] / t, v1.e[2] / t);
}

__host__ __device__ inline double dot(const vec3& v1, const vec3& v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
	return vec3(
		v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1], 
		-v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0],
		v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]
	);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v)
{
	return v / v.length(); 
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state)
{
	vec3 p;
	do
	{
		// currand_uniform return [0, 1]
		// 2.0 * [0, 1] - 1 makes [-1, 1]
		//p = 2.0 * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vec3(1, 1, 1);
		p = 2.0 * RANDVEC3 - vec3(1, 1, 1); 
	} while (p.squared_length() >= 1.0);
	return p; 
}

__device__ inline vec3 random_unit_sphere(curandState* local_rand_state)
{
	return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ inline vec3 random_in_hemisphere(curandState* local_rand_state, const vec3& normal)
{
	vec3 in_unit_sphere = random_unit_sphere(local_rand_state);
	if (dot(in_unit_sphere, normal) > 0.0)
		return in_unit_sphere;
	else 
		return -in_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3 n)
{
	return v - 2.0 * dot(v, n) * n; 
}

#endif
