# raytracer

My implementation of [Peter Shirley's Ray Tracing in one weekend](https://github.com/petershirley/raytracinginoneweekend/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The basic requirements for building the executable are:

* CMake 2.8+
* a c++ compiler
* make

#### Installation on Ubuntu

```
sudo apt-get install build-essentials cmake
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
Now, you should see the executable in the build folder.
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

## Authors

* **Biró Enikő** - [BiroEniko](https://github.com/biroeniko)
