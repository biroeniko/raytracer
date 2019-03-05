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

#include <OpenImageDenoise/oidn.hpp>

struct Image
{
    vec3* pixels;
    vec3* pixels2;

    uint32_t* windowPixels;
    uint8_t* fileOutputImage;

    const int nx;
    const int ny;
    const int tx;
    const int ty;

    bool showWindow;
    bool writeImage;

    oidn::DeviceRef device;
    oidn::FilterRef filter;

    CUDA_HOSTDEV Image(bool showWindow, bool writeImage, int x, int y, int tx, int ty ) : showWindow(showWindow), writeImage(writeImage), nx(x), ny(y), tx(tx), ty(ty)
    {
        #ifdef CUDA_ENABLED
            int pixelCount = nx*ny;
            size_t pixelsFrameBufferSize = pixelCount*sizeof(vec3);
            size_t windowPixelsFrameBufferSize = pixelCount*sizeof(uint32_t);
            size_t fileOutputImageFrameBufferSize = 3*pixelCount*sizeof(uint8_t);

            // allocate Frame Buffers
            checkCudaErrors(cudaMallocManaged((void **)&pixels, pixelsFrameBufferSize));
            checkCudaErrors(cudaMallocManaged((void **)&pixels2, pixelsFrameBufferSize));
            checkCudaErrors(cudaMallocManaged((void **)&windowPixels, windowPixelsFrameBufferSize));
            checkCudaErrors(cudaMallocManaged((void **)&fileOutputImage, fileOutputImageFrameBufferSize));
        #else
            pixels = new vec3[nx*ny];
            pixels2 = new vec3[nx*ny];

            if (showWindow)
                windowPixels = new uint32_t[nx*ny];

            if (writeImage)
                fileOutputImage = new uint8_t[nx * ny * 3];

        #endif // CUDA_ENABLED

        #ifdef OIDN_ENABLED
            // Create an Open Image Denoise device
            device = oidn::newDevice();
            device.commit();

            // Create a denoising filter
            filter = device.newFilter("RT"); // generic ray tracing filter
            filter.setImage("color", pixels2, oidn::Format::Float3, nx, ny);
            filter.setImage("output", pixels2, oidn::Format::Float3, nx, ny);
            filter.set("hdr", true); // image is HDR
            filter.commit();

        #endif // OIDN_ENABLED

    }

    #ifdef OIDN_ENABLED
        void denoise()
        {
            // Filter the image
            filter.execute();

            // Check for errors
            const char* errorMessage;
            if (device.getError(errorMessage) != oidn::Error::None)
                std::cout << "Error: " << errorMessage << std::endl;
        }
    #endif // OIDN_ENABLED

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
                pixels[i] = vec3(0, 0, 0);
            }
        #endif // CUDA_ENABLED
    }

    void savePfm()
    {
        FILE* f = fopen("wtf.pfm", "wb");
        fprintf(f, "PF\n%d %d\n-1\n", nx, ny);
        fwrite(pixels2, sizeof(float), nx*ny*3, f);
        fclose(f);
    }

    CUDA_HOSTDEV ~Image()
    {
        #ifdef CUDA_ENABLED
            checkCudaErrors(cudaFree(pixels));
            checkCudaErrors(cudaFree(pixels2));
            checkCudaErrors(cudaFree(windowPixels));
            checkCudaErrors(cudaFree(fileOutputImage));
        #else
            delete [] pixels;
            delete [] pixels2;

            if (showWindow)
                delete[] windowPixels;

            if (writeImage)
                delete[] fileOutputImage;
        #endif // CUDA_ENABLED
    }
};
