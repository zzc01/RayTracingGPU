#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"

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
__device__ bool hit_sphere(const point3& center, double radius, const ray& r)
{
	vec3 oc = r.origin() - center; 
	double a = dot(r.direction(), r.direction());
	double b = 2.0 * dot(oc, r.direction()); 
	double c = dot(oc, oc) - radius * radius;
	double discriminant = b * b - 4.0 * a * c; 
	return discriminant > 0.0; 
}

__device__ color3 ray_color(const ray& r)
{
	if (hit_sphere(vec3(0, 0, -1), 0.5, r))
		return color3(1, 0, 0);
	vec3 unit_direction = unit_vector(r.direction());
	double t = 0.5 * (unit_direction.y() + 1.0f); 
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y,
					   vec3 lower_left_corner, vec3 horizontal, 
					   vec3 vertical, vec3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	double u = double(i) / double(max_x); 
	double v = double(j) / double(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	fb[pixel_index] = ray_color(r);
}


int main()
{
	// Image 
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

	// profiling 
	clock_t time0, time1, time2; 
	time0 = clock();

	// Render 
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty); 
	render << <blocks, threads >> > (fb, nx, ny, 
									point3(-2.0, -1.0, -1.0),
									vec3(4.0, 0.0, 0.0),
									vec3(0.0, 2.0, 0.0),
									vec3(0.0, 0.0, 0.0) );
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


	// profiling 
	time2 = clock();
	timer_seconds = ((double)(time2 - time1)) / CLOCKS_PER_SEC;
	std::cerr << "Ouput took " << timer_seconds << " sec." << std::endl;

	std::cerr << "Done!" << std::endl;
}