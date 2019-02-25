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
#include "util/renderer.h"

#include "hitables/sphere.h"
#include "hitables/hitableList.h"
#include "util/camera.h"
#include "materials/material.h"
#include "util/scene.cuh"
#include "util/window.h"

const int numHitables = 102;

#ifdef CUDA_ENABLED
    void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable*** list, hitable** world, Window** w, Image** image, Camera** cam, Renderer** renderer)
    {
        // World
        checkCudaErrors(cudaMallocManaged(list, numHitables*sizeof(hitable*)));
        hitable** worldPtr;
        checkCudaErrors(cudaMallocManaged(&worldPtr, sizeof(hitable*)));
        randomScene2<<<1,1>>>(*list, worldPtr);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        *world = *worldPtr;
        checkCudaErrors(cudaFree(worldPtr));

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
            RandomGenerator rng(sampleCount * nsBatch + s, i*image->nx + j);
            float u = float(i + rng.get1f()) / float(image->nx); // left to right
            float v = float(j + rng.get1f()) / float(image->ny); // bottom to top
            ray r = cam->getRay(rng, u, v);

            image->pixels[pixelIndex] += render->color(rng, r, world, 0);
        }

        vec3 col = image->pixels[pixelIndex] / (sampleCount * nsBatch);

        // Gamma encoding of images is used to optimize the usage of bits
        // when encoding an image, or bandwidth used to transport an image,
        // by taking advantage of the non-linear manner in which humans perceive
        // light and color. (wikipedia)

        // we use gamma 2: raising the color to the power 1/gamma (1/2)
        col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

        int ir = int(255.99f*col[0]);
        int ig = int(255.99f*col[1]);
        int ib = int(255.99f*col[2]);

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
    void Renderer::cudaRender(uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage)
    {
        dim3 blocks( (image->nx + image->tx - 1)/image->tx, (image->ny + image->ty - 1)/image->ty);
        dim3 threads(image->tx, image->ty);

        render<<<blocks, threads>>>(cam, image, world, this, sampleCount);
        //std::cout << (image->nx + image->tx - 1)/image->tx;
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

    }
#endif // CUDA_ENABLED
