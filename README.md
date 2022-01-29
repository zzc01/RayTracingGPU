# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".
 
Chapter 7. Antialiasing <br />
  7.1  Some Random Number Utilities <br />
  7.2  Generating Pixels with Multiple Samples <br />
 <br />
Notes:  <br />
1. Random number generation in cuda  <br />
#include <curand_kernel.h> <br />
render_init << <blocks, threads >> > (nx, ny, d_rand_state); <br />
every pixel has a random state.  <br />
render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);  <br />
for each pixel, use that random state to generate 100 times the ray trace  <br />

2. cudaDeviceReset(); <br />
useful for cuda-memcheck --leak-check full?  <br />

3. The 100 number of samples later will be 100 number of ray tracing?  <br />
I feel like it should be possible to also multi thread the 100 number of samples  <br />


References:  <br />
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

