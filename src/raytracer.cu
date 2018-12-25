/* MIT License
Copyright (c) 2018 Biro Eniko
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hitables/sphere.h"
#include "hitables/hitableList.h"
#include "util/camera.h"
#include "materials/material.h"
#include "util/scene.h"
#include "util/renderer.h"
#include "util/window.h"
#include "util/common.h"

void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable** world, Window** w, Image** image, Camera** cam, Renderer** render)
{
    hitable** list;
    int num_hitables = 4;
    checkCudaErrors(cudaMallocManaged((void **)&list, num_hitables*sizeof(hitable *)));
    checkCudaErrors(cudaMallocManaged((void **)&world, sizeof(hitable *)));
    simpleScene<<<1,1>>>(list, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Camera *)));
    checkCudaErrors(cudaMallocManaged((void **)&image, sizeof(Image *)));
    checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(Renderer *)));
    *image = new Image(showWindow, writeImagePPM || writeImagePNG, nx, ny, tx, ty);

    vec3 lookFrom(13.0f, 2.0f, 3.0f);
    vec3 lookAt(0.0f, 0.0f, 0.0f);
    float distToFocus = 10.0f;
    float aperture = 0.1f;

    *cam = new Camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), distToFocus);
    *render = new Renderer(showWindow, writeImagePPM, writeImagePNG);

    *world = simpleScene2();

    if (showWindow)
    {
        checkCudaErrors(cudaMallocManaged((void **)&w, sizeof(Window *)));
        *w = new Window(*cam, *render, nx, ny, thetaInit, phiInit, zoomScale, stepScale);
    }

}

void destroyWorldCuda(bool showWindow, hitable* world, Window* w, Image* image, Camera* cam, Renderer* render)
{

}
