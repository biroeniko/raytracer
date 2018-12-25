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
        vec3* pixels;
    #else
        vec3** pixels;
    #endif // CUDA_ENABLED

    uint32_t* windowPixels;
    uint8_t* fileOutputImage;

    const int nx;
    const int ny;
    const int tx;
    const int ty;

    bool showWindow;
    bool writeImage;

    CUDA_HOSTDEV Image(bool showWindow, bool writeImage, int x, int y, int tx, int ty ) : showWindow(showWindow), writeImage(writeImage), nx(x), ny(y), tx(tx), ty(ty)
    {
        #ifdef CUDA_ENABLED
            int pixelCount = nx*ny;
            size_t pixelsFrameBufferSize = pixelCount*sizeof(vec3);
            size_t windowPixelsFrameBufferSize = pixelCount*sizeof(uint32_t);
            size_t fileOutputImageFrameBufferSize = 3*pixelCount*sizeof(uint8_t);

            // allocate Frame Buffers
            checkCudaErrors(cudaMallocManaged((void **)&pixels, pixelsFrameBufferSize));
            checkCudaErrors(cudaMallocManaged((void **)&windowPixels, windowPixelsFrameBufferSize));
            checkCudaErrors(cudaMallocManaged((void **)&fileOutputImage, fileOutputImageFrameBufferSize));
        #else
            pixels = new vec3*[nx];
            for (int i = 0; i < nx; i++)
                pixels[i] = new vec3[ny];

            if (showWindow)
                windowPixels = new uint32_t[nx*ny];

            if (writeImage)
                fileOutputImage = new uint8_t[nx * ny * 3];

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
            checkCudaErrors(cudaFree(pixels));
            checkCudaErrors(cudaFree(windowPixels));
            checkCudaErrors(cudaFree(fileOutputImage));
        #else
            for (int i = 0; i < nx; ++i)
                delete [] pixels[i];
            delete [] pixels;

            if (showWindow)
                delete[] windowPixels;

            if (writeImage)
                delete[] fileOutputImage;
        #endif // CUDA_ENABLED
    }
};
