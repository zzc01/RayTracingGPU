#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"

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

//// Render
__device__ color3 ray_color(const ray& r, hittable** world)
{
	// why using pointer to pointer 
	// this memory is in the gpu stack 
	hit_record rec; 
	if ((*world)->hit(r, 0.0, DBL_MAX, rec))
	{
		return 0.5 * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f); 
	}
	else
	{
		vec3 unit_direction = unit_vector(r.direction()); 
		double t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}
}

__global__ void render(vec3* fb, int max_x, int max_y,
					   vec3 lower_left_corner, vec3 horizontal, 
					   vec3 vertical, vec3 origin, hittable** world)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	double u = double(i) / double(max_x); 
	double v = double(j) / double(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	fb[pixel_index] = ray_color(r, world);
}

__global__ void create_world(hittable** d_list, hittable** d_world)
{
	// is it in purpose using pointer-to-pointer? 
	// d_ naming 
	// why detect 0 and 0 Idx 
	// are these in the GPU? 
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// this is to replace the vector<hitable> and append 
		*(d_list)	= new sphere(vec3(0, 0, -1), 0.5);
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world = new hittable_list(d_list, 2);
	}

}

// cast to void** for cudaMemallocManaged 
__global__ void free_world(hittable** d_list, hittable** d_world)
{
	delete* (d_list);
	delete* (d_list+1);
	delete* (d_world);
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

	// world 
	// these object are not accessed by the host. thus don't need to use cudaMallocManaged 
	// to me this looks like cudaMalloc gpu memory for the pointer 
	// and then in the device code malloc gpu memory for the ptr to ptr 
	// when return need to delete the malloc 
	// at the end need to free the cudaMalloc 
	hittable** d_list; 
	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
	hittable** d_world; 
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
	create_world << <1, 1 >> > (d_list, d_world); 
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render 
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty); 
	render << <blocks, threads >> > (fb, nx, ny, 
									point3(-2.0, -1.0, -1.0),
									vec3(4.0, 0.0, 0.0),
									vec3(0.0, 2.0, 0.0),
									vec3(0.0, 0.0, 0.0),
									d_world);
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
	free_world << <1, 1 >> > (d_list, d_world); 
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	// why previous did not do this 
	checkCudaErrors(cudaFree(fb));


	// profiling 
	time2 = clock();
	timer_seconds = ((double)(time2 - time1)) / CLOCKS_PER_SEC;
	std::cerr << "Ouput took " << timer_seconds << " sec." << std::endl;

	std::cerr << "Done!" << std::endl;
}