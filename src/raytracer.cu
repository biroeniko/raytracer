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

CUDA_GLOBAL void createWorld(bool showWindow, bool writeImagePPM, bool writeImagePNG, Camera** cam)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *cam = new Camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), distToFocus);
    }
}

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

    if (showWindow)
        checkCudaErrors(cudaMallocManaged((void **)&w, sizeof(Window *)));

    createWorld<<<1,1>>>(showWindow, writeImagePPM, writeImagePNG, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void destroyWorldCuda(bool showWindow, hitable* world, Window* w, Image* image, Camera* cam, Renderer* render)
{

}
