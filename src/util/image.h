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
#include "util/imagedenoiser.h"

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

    ImageDenoiser denoiser;

    CUDA_HOST Image(bool showWindow, bool writeImage,
                    int x, int y, int tx, int ty );

    void denoise()
    {
        #ifdef OIDN_ENABLED
            denoiser.denoise();
        #endif // OIDN_ENABLED
    }

    #ifdef CUDA_ENABLED
        void cudaResetImage();
    #endif // CUDA_ENABLED

    CUDA_HOSTDEV void resetImage();
    void savePfm();
    CUDA_HOST ~Image();

};
