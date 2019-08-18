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

#include "util/image.h"

CUDA_HOST Image::Image(bool showWindow, bool writeImage,
                       int x, int y, int tx, int ty ) :
                       nx(x), ny(y), tx(tx), ty(ty),
                       showWindow(showWindow),
                       writeImage(writeImage)
{
    #ifdef CUDA_ENABLED
        int pixelCount = nx*ny;
        size_t pixelsFrameBufferSize = static_cast<size_t>(pixelCount)*sizeof(vec3);
        size_t windowPixelsFrameBufferSize = static_cast<size_t>(pixelCount)*sizeof(uint32_t);
        size_t fileOutputImageFrameBufferSize = static_cast<size_t>(3*pixelCount)*sizeof(uint8_t);

        // allocate Frame Buffers
        checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&pixels), pixelsFrameBufferSize));
        checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&pixels2), pixelsFrameBufferSize));
        checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&windowPixels), windowPixelsFrameBufferSize));
        checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fileOutputImage), fileOutputImageFrameBufferSize));
    #else
        pixels = new vec3[nx*ny];
        pixels2 = new vec3[nx*ny];

        if (showWindow)
            windowPixels = new uint32_t[nx*ny];

        if (writeImage)
            fileOutputImage = new uint8_t[nx * ny * 3];

    #endif // CUDA_ENABLED

    ImageDenoiser denoiserForPixels(pixels2, nx, ny);
    denoiser = denoiserForPixels;

}

CUDA_HOSTDEV void Image::resetImage()
{
    #ifdef CUDA_ENABLED
        cudaResetImage();
    #else
        #pragma omp parallel for
        for (int i = 0; i < nx*ny; i++)
        {
            pixels[i] = vec3(0.0f, 0.0f, 0.0f);
        }
    #endif // CUDA_ENABLED
}

void Image::savePfm()
{
    FILE* f = fopen("wtf.pfm", "wb");
    fprintf(f, "PF\n%d %d\n-1\n", nx, ny);
    fwrite(pixels2, sizeof(float), static_cast<size_t>(nx*ny*3), f);
    fclose(f);
}

CUDA_HOST Image::~Image()
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

