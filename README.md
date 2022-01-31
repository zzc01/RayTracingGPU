# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".

 
Chapter 10. Dielectrics<br/>
  10.1  Refraction<br/>
  10.2  Snell's Law<br/>
  10.3  Total Internal Reflection<br/>
  10.4  Schlick Approximation<br/>
  10.5  Modeling a Hollow Glass Sphere<br/>
  
  
Notes:<br/>
1. hit_rec in hittable is a scoped object, is generated in device. Because it is scoped it will be destroyed when out of scope.<br/> 
2. On the other hand, the world, d_list, sphere, material, metal, lambertian, dieletric. Alghouth they are device object, but they need to survive after out of scope. Thus they are defined using new and held by pointers.<br/>  
3. In many if() conditions forgot to return false. But in cuda does not give warning?<br/> 
4. Made a lot of mistake on the set_face_normal() method in hit_record struct. Ex. outward_normal and bool front_face.<br/>  
5. Profiling:<br/>
a) 1100 msec<br/>
b) 1563 msec<br/>
c) 1705 msec<br/> 
d) 2358 msec<br/> 
 
 
References:<br/>
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

