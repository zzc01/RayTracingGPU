# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".

 
Chapter 9.  Metal<br/>
  9.1  An Abstract Class for Materials<br/>
  9.2  A Data Structure to Describe Ray-Object Intersections<br/>
  9.3  Modeling Light Scatter and Reflectance<br/>
  9.4  Mirrored Light Reflection<br/>
  9.5  A Scene with Metal Spheres<br/>
  9.6  Fuzzy Reflection<br/>
  
  
Notes:<br/>
1. add material parent class. And children classes lambertian, metal.<br/>
2. add reflect and scatter concept into these material classes.<br/> 
3. add mat_ptr to hit_record, sphere object classes<br/>  
4. New way to define d_list<br/> 
d_list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(color3(0.8, 0.3, 0.3)));<br/>
d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(color3(0.8, 0.8, 0.0)));<br/>
5. 
for (int i = 0; i < 4; i++)
{
	// d_list is a hittable ptr. suppose it does not have mat_ptr. 
	// thus to convert to sphere ptr ... this is pretty not manual not common. 
	delete ((sphere*)d_list[i])->mat_ptr;
	delete d_list[i];
}
6. The cross #include can be complicated ?<br/>
in hittable.h <br/>
//#include "ray.h"<br/>
//#include "material.h"<br/>
class material;<br/>
in material.h <br/>
struct hit_record; <br/>
#include "ray.h"<br/>
//#include "hittable.h"<br/>

 
 
References:  <br />
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

