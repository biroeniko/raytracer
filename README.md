# raytracer

My implementation of [Peter Shirley's Ray Tracing in one weekend](https://github.com/petershirley/raytracinginoneweekend/).

## Some versions are available through different branches:
* [CPU version, accelerated by OpenMP](https://github.com/biroeniko/raytracer/tree/cpu-only)

## Details:
This code follows the steps of Peter Shirley's mini book series.
### Features implemented so far are:
* Cmake support
* Basic implementation of all features included in [Peter Shirley's Ray Tracing in one weekend](https://github.com/petershirley/raytracinginoneweekend/)
* Multithreaded implementation with OpenMP
* Supported output formats: PNG with [STB image library](https://github.com/nothings/stb) and PPM
* [SDL2](https://www.libsdl.org/) for real-time display support + keyboard movement support
* [PCG](http://www.pcg-random.org/) random number generator (the minimal C implementation)

### Features to be implemented:
* Other two books
* Accelerated versions
  * CUDA
  * ispc
  * use denoising

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The basic requirements for building the executable are:

* CMake 2.8+
* a C++ compiler
* make
* SDL2
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
![](https://github.com/biroeniko/raytracer/blob/master/images/final.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/final2.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/final3.png)

Some screenshots along the way of developing:
![](https://github.com/biroeniko/raytracer/blob/master/images/sphereHit.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/interesting.png)
![](https://github.com/biroeniko/raytracer/blob/master/images/test.png)
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

## Authors

* **Biró Enikő** - [BiroEniko](https://github.com/biroeniko)
