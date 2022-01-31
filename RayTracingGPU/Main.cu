#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include <curand_kernel.h>
#include "material.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

//cudaError_t result,
void check_cuda(int result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' " << std::endl;
		// Make sure to call CUDA device reset before exist. For profiling tool to work.  
		cudaDeviceReset();
		exit(99);
	}
}

// Render
__device__ color3 ray_color(const ray& r, hittable** world, curandState *local_rand_state)
{
	ray cur_ray = r; 
	point3 cur_attenuation = point3(1.0, 1.0, 1.0); 
	for (int i = 0; i < 50; i++)
	{
		hit_record rec;
		if ((*world)->hit(cur_ray, 1e-10, DBL_MAX, rec))
		{
			ray scattered; 
			point3 attenuation; 
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else
			{
				return color3(0.0, 0.0, 0.0); 
			}
		}
		else
		{
			vec3 unit_direction = unit_vector(cur_ray.direction());
			double t = 0.5 * (unit_direction.y() + 1.0);
			vec3 color = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0); 
			return cur_attenuation * color;
		}
	}
	// exceeded recussion depth 
	return vec3(0.0, 0.0, 0.0); 
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	// Each thread gets same seed, a different sequence number, no offset 
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y,
					   int ns, camera** cam, hittable** world, 
					   curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++)
	{
		double u = double(i + curand_uniform(&local_rand_state)) / double(max_x);
		double v = double(j + curand_uniform(&local_rand_state)) / double(max_y);
		ray r = (*cam)->get_ray(u, v); 
		col += ray_color(r, world, &local_rand_state);
	}
	// The state value will change? 
	rand_state[pixel_index] = local_rand_state; 
	col /= double(ns); 
	col[0] = sqrt(col[0]); 
	col[1] = sqrt(col[1]); 
	col[2] = sqrt(col[2]); 
	fb[pixel_index] = col; 
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// this is to replace the vector<hitable> and append 
		d_list[0] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(color3(0.8, 0.8, 0.0)));
		d_list[1] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(color3(0.1, 0.2, 0.5)));
		d_list[2] = new sphere(vec3(-1.0, 0.0, -1), 0.5, new dieletric(1.5));
		d_list[3] = new sphere(vec3(-1.0, 0.0, -1), -0.45, new dieletric(1.5));
		d_list[4] = new sphere(vec3(1.0, 0.0, -1), 0.5, new metal(color3(0.8, 0.6, 0.2), 0.0));
		*d_world = new hittable_list(d_list, 5);
		*d_camera = new camera(	point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0), 90.0, double(nx)/double(ny));
	}
}

// cast to void** for cudaMemallocManaged 
__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera)
{
	for (int i = 0; i < 5; i++)
	{
		// d_list is a hittable ptr. suppose it does not have mat_ptr. 
		// this requires to convert to sphere ptr ... this is pretty not manual 
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* (d_world);
	delete* (d_camera);
}


int main()
{
	// profiling 
	clock_t time0, time1, time2;
	time0 = clock();

	// Image 
	int ns = 100; 
	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks" << std::endl;

	int num_pixels = nx * ny;

	// allocate FB 
	color3* fb;
	size_t fb_size = num_pixels * sizeof(color3);
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random state 
	curandState* d_rand_state; 
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	// world 
	hittable** d_list; 
	checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(hittable*)));
	hittable** d_world; 
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny); 
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render 
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty); 
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state); 
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// profiling 
	time1 = clock();
	double timer_seconds = ((double)(time1 - time0)) ;
	std::cerr << "Cuda took " << timer_seconds << " msec." << std::endl;


	// Output FB as Image 
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t pixel_index = (j * nx + i);
			int ir = static_cast<int>(255.99 * fb[pixel_index].r());
			int ig = static_cast<int>(255.99 * fb[pixel_index].g());
			int ib = static_cast<int>(255.99 * fb[pixel_index].b());
			std::cout << ir << ' ' << ig << ' ' << ib << std::endl; 
		}
	}

	//clean up 
	checkCudaErrors(cudaDeviceSynchronize()); 
	free_world << <1, 1 >> > (d_list, d_world, d_camera); 
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_rand_state));
	// why previous did not do this 
	checkCudaErrors(cudaFree(fb));

	// useful for cuda-memcheck --leak-check full
	cudaDeviceReset();

	// profiling 
	time2 = clock();
	timer_seconds = ((double)(time2 - time1)) / CLOCKS_PER_SEC;
	std::cerr << "Ouput took " << timer_seconds << " sec." << std::endl;

	std::cerr << "Done!" << std::endl;
}