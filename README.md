# RayTracingGPU
A Ray Tracing Practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".
 
Chapter 7. Antialiasing
  7.1  Some Random Number Utilities
  7.2  Generating Pixels with Multiple Samples


Notes: 
1. Random number generation in cuda 
#include <curand_kernel.h>
render_init << <blocks, threads >> > (nx, ny, d_rand_state);
every pixel has a random state. 
render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state); 
for each pixel, use that random state to generate 100 times the ray trace 

2. cudaDeviceReset();
useful for cuda-memcheck --leak-check full? 

3. The 100 number of samples later will be 100 number of ray tracing? 
I feel like it should be possible to also multi thread the 100 number of samples 


References: 
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

