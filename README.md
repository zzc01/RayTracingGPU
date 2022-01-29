# RayTracingGPU
A Ray Tracing Practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".
 
Chapter 6.  Surface Normals and Multiple Objects
  6.1  Shading with Surface Normals
  6.2  Simplifying the Ray-Sphere Intersection Code
  6.3  An Abstraction for Hittable Objects
  6.4  Front Faces Versus Back Faces
  6.5  A List of Hittable Objects
  6.6  Some New C++ Features
  6.7  Common Constants and Utility Functions


Notes: 
1. Doing new and delete in __device__. create_world() and free_world()
2. Using both cudaMalloc() and cudaMallocManaged() 
3. Using cudaFree 
4. Using ptr to ptr. Is this to access the object in device? why use ptr to ptr 
5. Did not implement the front_face to detect inner or outer 
6. Not using vector<hittable> and append sphere any more 
 use a hittable** list and hittable** world 
  

References: 
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

