#pragma once
#ifndef HITTABLE_LISTH
#define HITTABLE_LISTH

#include "hittable.h"

class hittable_list : public hittable
{
public: 
	__device__ hittable_list(){}
	__device__ hittable_list(hittable** l, int n): list(l), list_size(n) {}
	__device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;
	hittable** list;
	int list_size; 
};


__device__ bool hittable_list::hit(const ray& r, double tmin, double tmax, hit_record& rec) const
{
	hit_record temp_rec; 
	bool hit_anything = false; 
	float closest_so_far = tmax; 
	for (int i = 0; i < list_size; i++)
	{
		if (list[i]->hit(r, tmin, closest_so_far, temp_rec))
		{
			hit_anything = true; 
			closest_so_far = temp_rec.t; 
			rec = temp_rec; 
		}
	}
	return hit_anything; 
}

#endif 