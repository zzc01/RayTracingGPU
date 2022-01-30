# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".

 
Chapter 8.  Diffuse Materials<br />
  8.1  A Simple Diffuse Material<br />
  8.2  Limiting the Number of Child Rays<br />
  8.3  Using Gamma Correction for Accurate Color Intensity<br />
  8.4  Fixing Shadow Acne<br />
  8.5  True Lambertian Reflection<br />
  8.6  An Alternative Diffuse Formulation<br />


Notes:<br />
1. The recurssive part was writen in iteration way. Can GPU Cuda do recurssion? 
This requires to use stack memory when going deeper in the recussion. Saw online 
only special GPU can do recurssion. Saw another discussion using template <depth>
to do the recussion, quite smart.<br /> 

2. The random vector<br /> 
__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state)<br />
cannot do __host__ because of the curand_uniform() called inside the function<br /> 

3. The reason for ptr to ptr for class object in the device<br /> 
Is because of class object. If it were int or curandState use ptr.<br /> 
But it is a class object use ptr to ptr... this could help of the use of parent-children virtual function.<br /> 
color3* fb; the real intention is to build a number of color3 object.<br />  
hittable** d_list; this could built a number of hittable* ptr ... for later use parent-children virtual function.<br /> 
 
 
References:  <br />
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

