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

#ifdef OIDN_ENABLED
    #include <OpenImageDenoise/oidn.hpp>
#endif // OIDN_ENABLED

struct ImageDenoiser
{

    oidn::DeviceRef device;
    oidn::FilterRef filter;

    CUDA_HOST ImageDenoiser()
    {

    }

    CUDA_HOST ImageDenoiser(vec3* pixels, int nx, int ny)
    {
        // Create an Open Image Denoise device
        device = oidn::newDevice();
        device.commit();

        // Create a denoising filter
        filter = device.newFilter("RT"); // generic ray tracing filter
        filter.setImage("color", pixels, oidn::Format::Float3, static_cast<size_t>(nx), static_cast<size_t>(ny));
        filter.setImage("output", pixels, oidn::Format::Float3, static_cast<size_t>(nx), static_cast<size_t>(ny));
        filter.set("hdr", true); // image is HDR
        filter.commit();
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

};
