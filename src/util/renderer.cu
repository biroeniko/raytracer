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

#include "util/common.h"
#include "util/globals.h"
#include "util/renderer.h"

#include "hitables/sphere.h"
#include "hitables/hitableList.h"
#include "util/camera.h"
#include "materials/material.h"
#include "util/scene.cuh"
#include "util/window.h"

CUDA_DEV int numHitables = 0;

#ifdef CUDA_ENABLED
    void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable*** list, hitable** world, Window** w, Image** image, Camera** cam, Renderer** renderer)
    {
        int choice = 4;

        switch(choice)
        {
            case 0:
                numHitables = 4;
                break;
            case 1:
                numHitables = 58;
                break;
            case 2:
                numHitables = 901;
                break;
            case 3:
                numHitables = 102;
                break;
            case 4:
                numHitables = 68;
                break;
        }

        // World
        checkCudaErrors(cudaMallocManaged(list, numHitables*sizeof(hitable*)));
        hitable** worldPtr;
        checkCudaErrors(cudaMallocManaged(&worldPtr, sizeof(hitable*)));
        switch(choice)
        {
            case 0:
                simpleScene<<<1,1>>>(*list, worldPtr);
                break;
            case 1:
                simpleScene2<<<1,1>>>(*list, worldPtr);
                break;
            case 2:
                randomScene<<<1,1>>>(*list, worldPtr);
                break;
            case 3:
                randomScene2<<<1,1>>>(*list, worldPtr);
                break;
            case 4:
                randomScene3<<<1,1>>>(*list, worldPtr);
                break;
        }
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        *world = *worldPtr;
        checkCudaErrors(cudaFree(worldPtr));

        // Camera
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
    }

    CUDA_GLOBAL void freeWorldCuda(hitable** list, hitable** world)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            for (int i = 0; i < numHitables; i++)
            {
                delete ((sphere *)list[i])->matPtr;
                delete list[i];
            }
            //delete *world;
        }
    }

    void destroyWorldCuda(bool showWindow, hitable** list, hitable* world, Window* w, Image* image, Camera* cam, Renderer* render)
    {
        freeWorldCuda<<<1,1>>>(list, &world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(cam));
        checkCudaErrors(cudaFree(render));
        checkCudaErrors(cudaFree(image));
    }

    CUDA_GLOBAL void render(Camera* cam, Image* image, hitable* world, Renderer* render, int sampleCount)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= image->nx) || (j >= image->ny))
            return;

        int pixelIndex = j*image->nx + i;

        // Render the samples in batches
        for (int s = 0; s < nsBatch; s++)
        {
            RandomGenerator rng(sampleCount * nsBatch + s, pixelIndex);
            float u = float(i + rng.get1f()) / float(image->nx); // left to right
            float v = float(j + rng.get1f()) / float(image->ny); // bottom to top
            ray r = cam->getRay(rng, u, v);

            image->pixels[pixelIndex] += render->color(rng, r, world, 0);
        }

        vec3 col = image->pixels[pixelIndex] / (sampleCount * nsBatch);

        image->pixels2[pixelIndex] = col;
    }

    CUDA_GLOBAL void display(Image* image)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int pixelIndex = j*image->nx + i;

        vec3 col = image->pixels2[pixelIndex];

        // Gamma encoding of images is used to optimize the usage of bits
        // when encoding an image, or bandwidth used to transport an image,
        // by taking advantage of the non-linear manner in which humans perceive
        // light and color. (wikipedia)

        // we use gamma 2: raising the color to the power 1/gamma (1/2)
        col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

        int ir = clamp(int(255.f*col[0]), 0, 255);
        int ig = clamp(int(255.f*col[1]), 0, 255);
        int ib = clamp(int(255.f*col[2]), 0, 255);

        if (image->writeImage)
        {
            // PNG
            int index = (image->ny - 1 - j) * image->nx + i;
            int index3 = 3 * index;

            image->fileOutputImage[index3 + 0] = ir;
            image->fileOutputImage[index3 + 1] = ig;
            image->fileOutputImage[index3 + 2] = ib;
        }

        if (image->showWindow)
            image->windowPixels[(image->ny-j-1)*image->nx + i] = (ir << 16) | (ig << 8) | (ib);
    }

#endif // CUDA_ENABLED

#ifdef CUDA_ENABLED
    void Renderer::cudaRender(Camera* cam, hitable* world, Image* image, int sampleCount)
    {
        dim3 blocks( (image->nx + image->tx - 1)/image->tx, (image->ny + image->ty - 1)/image->ty);
        dim3 threads(image->tx, image->ty);

        // Kernel call for the computation of pixel colors.
        render<<<blocks, threads>>>(cam, image, world, this, sampleCount);

        // Denoise here.
        #ifdef OIDN_ENABLED
            checkCudaErrors(cudaDeviceSynchronize());
            image->denoise();
            checkCudaErrors(cudaDeviceSynchronize());
        #endif // OIDN_ENABLED

        // Kernel call to fill the output buffers.
        display<<<blocks, threads>>>(image);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

    }
#endif // CUDA_ENABLED
