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

#pragma once

#include "util/vec3.h"
#include "util/util.h"

struct Image
{
    #ifdef CUDA_ENABLED
        vec3* frameBuffer;
    #else
        vec3** pixels;
    #endif // CUDA_ENABLED

    uint32_t *windowPixels;

    const int nx;
    const int ny;
    const int tx;
    const int ty;

    CUDA_HOSTDEV Image(int x, int y, int tx, int ty) : nx(x), ny(y), tx(tx), ty(ty)
    {
        #ifdef CUDA_ENABLED
            int pixelCount = nx*ny;
            size_t frameBufferSize = pixelCount*sizeof(vec3);
            // allocate Frame Buffer
            checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));
        #else
            pixels = new vec3*[nx];
            for (int i = 0; i < nx; i++)
                pixels[i] = new vec3[ny];

            windowPixels = new uint32_t[nx*ny];
        #endif // CUDA_ENABLED

    }

    #ifdef CUDA_ENABLED
        void cudaResetImage();
    #endif // CUDA_ENABLED

    CUDA_HOSTDEV void resetImage()
    {
        #ifdef CUDA_ENABLED
            cudaResetImage();
        #else
            #pragma omp parallel for
            for (int i = 0; i < nx*ny; i++)
            {
                pixels[i/ny][i%ny] = vec3(0, 0, 0);
            }
        #endif // CUDA_ENABLED
    }

    CUDA_HOSTDEV ~Image()
    {
        #ifdef CUDA_ENABLED
            checkCudaErrors(cudaFree(frameBuffer));
        #else
            for (int i = 0; i < nx; ++i)
                delete [] pixels[i];
            delete [] pixels;

            delete[] windowPixels;
        #endif // CUDA_ENABLED
    }
};
