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
        vec3 *frameBuffer;
    #else
        vec3** pixels;
    #endif // CUDA_ENABLED
    int rows;
    int columns;

    CUDA_HOSTDEV Image(int x, int y) : rows(x), columns(y)
    {
        #ifdef CUDA_ENABLED
            int pixelCount = x*y;
            size_t frameBufferSize = pixelCount*sizeof(vec3);
            // allocate Frame Buffer
            checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, frameBufferSize));
        #else
            pixels = new vec3*[rows];
            for (int i = 0; i < rows; i++)
                pixels[i] = new vec3[columns];
        #endif // CUDA_ENABLED

    }

    CUDA_HOSTDEV void resetImage()
    {
        #ifdef CUDA_ENABLED

        #else
        #pragma omp parallel for
        for (int i = 0; i < rows*columns; i++)
        {
            pixels[i/rows][i%columns] = vec3(0, 0, 0);
        }
        #endif // CUDA_ENABLED
    }

    CUDA_HOSTDEV ~Image()
    {
        #ifdef CUDA_ENABLED
            checkCudaErrors(cudaFree(frameBuffer));
        #else
            for (int i = 0; i < rows; ++i)
                delete [] pixels[i];
            delete [] pixels;
        #endif // CUDA_ENABLED
    }
};
