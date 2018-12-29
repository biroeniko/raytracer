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

#ifdef CUDA_ENABLED
    CUDA_GLOBAL void render(Camera* cam, Image* image, hitable** world, Renderer* render, int sampleCount)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= image->nx) || (j >= image->ny))
            return;

        RandomGenerator rng(sampleCount, i*image->nx + j);
        float u = float(i + rng.get1f()) / float(image->nx); // left to right
        float v = float(j + rng.get1f()) / float(image->ny); // bottom to top

        int pixelIndex = j*image->nx + i;

        ray r = cam->getRay(rng, u, v);

        //render->color(rng, r, world, 0);
        image->pixels[pixelIndex] += render->color(rng, r, world, 0);


    }
#endif // CUDA_ENABLED

void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable*** list, hitable*** world, Window** w, Image** image, Camera** cam, Renderer** renderer)
{
    hitable** list2;
    hitable** world2;
    // World
    int num_hitables = 4;
    checkCudaErrors(cudaMalloc(&list2, num_hitables*sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&world2, sizeof(hitable*)));
    simpleScene<<<1,1>>>(list2, world2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc(list, num_hitables*sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(world, sizeof(hitable*)));
    simpleScene<<<1,1>>>(*list, *world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Camera
    vec3 lookFrom(13.0f, 2.0f, 3.0f);
    vec3 lookAt(0.0f, 0.0f, 0.0f);
    checkCudaErrors(cudaMallocManaged(cam, sizeof(Camera)));
    new (*cam) Camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), distToFocus);

    // Renderer
    checkCudaErrors(cudaMallocManaged(renderer, sizeof(Renderer)));
    new (*renderer) Renderer(showWindow, writeImagePPM, writeImagePNG);

    // Image
    checkCudaErrors(cudaMallocManaged(image, sizeof(Image)));
    new (*image) Image(showWindow, writeImagePPM || writeImagePNG, nx, ny, tx, ty);

    // Window
    if (showWindow)
        *w = new Window(*cam, *renderer, nx, ny, thetaInit, phiInit, zoomScale, stepScale);

/*
    dim3 blocks((*image)->nx/(*image)->tx+1, (*image)->ny/(*image)->ty+1);
    dim3 threads((*image)->tx, (*image)->ty);

    render<<<blocks, threads>>>(*cam, *image, world, *renderer, 1);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    */

}

void destroyWorldCuda(bool showWindow, hitable*** list, hitable*** world, Window* w, Image* image, Camera* cam, Renderer* render)
{
    //checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(render));
    checkCudaErrors(cudaFree(image));
}
