# raytracer

My implementation of [Peter Shirley's Ray Tracing in one weekend](https://github.com/petershirley/raytracinginoneweekend/).

## Some versions are available through different branches:
* [CPU version, accelerated with OpenMP](https://github.com/biroeniko/raytracer/tree/cpu-only)
* [GPU+CPU version, accelerated with CUDA](https://github.com/biroeniko/raytracer/tree/gpu-cpu-first-book)
* [GPU+CPU version, accelerated with CUDA, with Open Image Denoise support](https://github.com/biroeniko/raytracer/tree/gpu-cpu-oidn-first-book)

## Details:
This code follows the steps of Peter Shirley's mini book series.
### Features implemented so far are:
* Cmake support
* Multithreaded implementation with OpenMP
* Supported output formats: PNG with [STB image library](https://github.com/nothings/stb) and PPM
* [SDL2](https://www.libsdl.org/) for real-time display support + keyboard movement support
* [CUDA](https://developer.nvidia.com/cuda-zone) support
* [Open Image Denoise](https://openimagedenoise.github.io/) support

* Implementation of all features included in [Peter Shirley's Ray Tracing In One Weekend](https://github.com/RayTracing/raytracing.github.io/blob/master/books/RayTracingInOneWeekend.html)
  * Spheres
  * Surface Normals
  * Antialiasing
  * Diffuse Materials
  * Metal
  * Dielectrics
  * Positionable Camera
  * Defocus Blur (“depth of field”)
* Implementation of the following features included in [Peter Shirley's Ray Tracing: The Next Week](https://github.com/RayTracing/raytracing.github.io/blob/master/books/RayTracingInOneWeekend.html)
  * Bounding volume hierarchy (BVH) support on CPU
  * Solid texture
  * Perlin noise support on CPU

### Features to be implemented:
* Ray Tracing: The Next Week
* Ray Tracing: The Rest of Your Life
* Bounding volume hierarchy (BVH) support on GPU

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The basic requirements for building the executable are:

* CMake 2.8+
* a C++ compiler
* make
* SDL2

Other requirements:
* CUDA support

#### Installation on Ubuntu

```
sudo apt-get install build-essentials cmake
sudo apt-get install libsdl2-dev
```

### Installing

This program have been tested on Ubuntu 16.04 but should work under any systems that fulfills the aforementioned requirements.

#### Installation on Ubuntu

If you succesfully cloned the source files and you are currently in the project directory, you can generate the Makefile using the following command:

```
mkdir build
cd build/
cmake ..
```
Now you can build the binary by running:

```
make
```
Now, you should see the executable in the build folder. Examples for the final images are:
![](https://github.com/biroeniko/raytracer/blob/master/images/cuda_640_360.gif)
![](https://github.com/biroeniko/raytracer/blob/master/images/noisy.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/denoised.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/final.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/final2.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/final3.png)

Some screenshots along the way of developing:
![](https://github.com/biroeniko/raytracer/blob/master/images/sphereHit.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/interesting.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/notAntialised.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/antialiased.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/diffuseBeforeGammaCorrection.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/diffuseWithGammaCorrection.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/diffuseWithGammaCorrectionAndShadowAcneCorrection.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/metal.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/metalWithFuzziness.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/hollowGlassSphere.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/cameraPosition.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/cameraPosition2.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/depthOfField.png)

## Built With

* [SDL2](https://www.libsdl.org/) - used for display
* [OpenMP](https://www.openmp.org/) - used for creating multiple threads for the tasks (pixel color calculation)
* [CUDA](https://developer.nvidia.com/cuda-zone) - used for acceleration
* [Open Image Denoise](https://openimagedenoise.github.io/) - used for acceleration

## Authors

* **Biró Enikő** - [BiroEniko](https://github.com/biroeniko)
