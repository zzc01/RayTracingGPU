# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".

 
Chapter 12. Defocus Blur<br/>
  12.1  A Thin Lens Approximation<br/>
  12.2  Generating Sample Rays<br/>
  
  
Notes:<br/>
1. Can understand the random unit disk and lens_radius concept. But cannot understand why at the focal_dist which equal to from - to in this chapter. Will be the most less blur distance? <br/>
a) If remove blur and change the focal_dist the picture does not change at all. <br/> 
b) Distance of focus (DoF). If off will take in more points' light, instead of focusing on the point's light. <br/>
2. Viewport distance is defined by focal_dist. If the object is on the same plan as viewport, when sweeping ray trace the object hit point equals to hit point on viewport. Though because added an offset the ray angle is different than from origin one. This does not affect blur. Can be viewed as scattering. <br/>
3. But if the object is not on the same plane as viewport, the hit point of the object is different. Thus basically not looking at the object. The more off from the viewport more distance the hitpoint. Thus more blur. <br/>
4. Profiling:<br/>
1903 msec<br/>  
 
 
References:<br/>
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

