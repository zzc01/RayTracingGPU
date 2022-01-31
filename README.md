# RayTracingGPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" and "Accelerated Ray Tracing in One Weekend in CUDA".

 
Chapter 11. Positionable Camera<br/>
  11.1  Camera Viewing Geometry<br/>
  11.2  Positioning and Orienting the Camera<br/>
  
  
Notes:<br/>
1. Need to define M_PI.<br/>
2. double half_height = tan(theta / 2);  This assumes viewport distance is 1 and 
the viewport height, width, lower_left_corner are all calculated accordingly<br/>
3. The vec3 cross function written wrong. Thus starting from ch02 to here was wrong. This causes the picutre angle too high and the distance shows to be too close.<br/> 
4. Profiling:<br/>
a) 780 msec <br/>
b) 2720 msec <br/>
 
 
References:<br/>
1) https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

