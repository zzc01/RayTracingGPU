#include <iostream>
#include <time.h>

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
__global__ void render(float* fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = 3 * (j * max_x + i);
	fb[pixel_index + 0] = double(i) / max_x;
	fb[pixel_index + 1] = double(j) / max_y;
	fb[pixel_index + 2] = 0.25;
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
	size_t fb_size = 3 * num_pixels * sizeof(float);

	// allocate FB 
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
	clock_t start, stop; 
	start = clock();

	// Render 
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty); 
	render << <blocks, threads >> > (fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << "sec." << std::endl; 


	// Output FB as Image 
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t pixel_index = 3 * (j * nx + i);
			double r = fb[pixel_index + 0];
			double g = fb[pixel_index + 1];
			double b = fb[pixel_index + 2];
			int ir = static_cast<int>(255.99 * r);
			int ig = static_cast<int>(255.99 * g);
			int ib = static_cast<int>(255.99 * b);

			std::cout << ir << ' ' << ig << ' ' << ib << std::endl; 
		}
	}

	std::cerr << "Done!" << std::endl;
}