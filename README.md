# Ray Tracing with GPU
A Ray Tracing practice project using GPU to speed up the rendering time. Following "Ray Tracing in One Weekend" [1] and "Accelerated Ray Tracing in One Weekend in CUDA" [2]. This is the ray tracing results of Chapter 13. You can navigate to different chapter's ray tracing results by switching to other branches. 
<br/><br/>

***
 
#### Chapter 13.  Where Next?<br/>
  13.1  A Final Render<br/>
  13.2  Next Steps<br/>


#### Notes:<br/>

In the final render we can compare the ray tracing time with using GPU and without using GPU. As you can see from the table below, using GPU speeds up the time by about 20 times. 
 
<br/>

| Tables        | Samples-per-pixel  |Time       |
| ------------- |:------------------:|:---------:|
| Using GPU     | 100                | 0.72 min  |
| Using GPU     | 500                | 3.57 min  |
| Without GPU   | 100                | 13.12 min |
| Without GPU   | 500                | 63.60 min |

<br/>

1. The curand_init(). <br/>
\_\_device\_\_ void curand_init ( unsigned long long seed, unsigned long long sequence, unsigned long long offset, curandState_t *state) <br/>

2. The pointer for the curandState. <br/>
// Pass by pointer is like telling children the address. But does this work if want to bring address information back? <br/>
// Here it works because the memory and address is allocated in main(). Thus later operations are operating on the address. <br/>
// Pass by reference is like aliasing. <br/>
curandState* d_rand_state; <br/>
checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));<br/>
\_\_global\_\_ void render_init(int max_x, int max_y, curandState* rand_state)<br/>
\_\_global\_\_ void rand_init(curandState *rand_state)<br/>


3. curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]); <br/>
// Each thread gets same seed, a different sequence number, no offset <br/>
// Original: Each thread gets same seed, a different sequence number, no offset<br/>
// curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);<br/>
// BUGFIX, see Issue#2: Each thread gets different seed, same sequence for<br/>
// performance improvement of about 2x! .... but did not improve for me <br/>

4. Why passing &local_rand_state? Because passed the value to local_rand_state. But why do this? this seems like if the state value is changed in curand_uniform() 
the value would not propagated back to the original space. No wonder done a re-assignment later. <br/>
\_\_global\_\_ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny, curandState *rand_state)<br/>
curandState local_rand_state = *rand_state; <br/>
#define RND (curand_uniform(&local_rand_state))<br/>
*rand_state = local_rand_state; <br/>

5. Profiling: <br/>
a) 1200x600, ns=100, 43017 msec <br/>
b) 1200x600, ns=500, 213971 msec <br/>

6. Profiling w/o GPU: <br/>
b) 1200x600, ns=100, 787410ms (787.41sec)<br/>
c) 1200x600, ns=500, Timer took 3.81613e+06ms (3816.13sec)	<br/>
 
 
# References:<br/>
1) Peter Shirley, https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview
2) Roger Allen, https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

