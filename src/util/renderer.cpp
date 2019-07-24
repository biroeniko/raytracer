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

#include "util/renderer.h"
#include "util/globals.h"
#include "util/scene.h"

#ifdef CUDA_ENABLED

#else
    CUDA_HOSTDEV void Renderer::render(int i, int j, std::unique_ptr<Camera>& cam,
                                       std::unique_ptr<Image>& image, hitable* world, int sampleCount)
    {
        int pixelIndex = j*nx + i;

        // Render the samples in batches
        for (int s = 0; s < nsBatch; s++)
        {
            RandomGenerator rng(sampleCount * nsBatch + s, pixelIndex);
            float u = float(i + rng.get1f()) / float(image->nx); // left to right
            float v = float(j + rng.get1f()) / float(image->ny); // bottom to top
            ray r = cam->getRay(rng, u, v);

            image->pixels[pixelIndex] += color(rng, r, world, 0);
        }

        vec3 col = image->pixels[pixelIndex] / sampleCount;

        image->pixels2[pixelIndex] = col;
    }

    CUDA_HOSTDEV void Renderer::display(int i, int j, std::unique_ptr<Image>& image)
    {
        int pixelIndex = j*image->nx + i;

        vec3 col = image->pixels2[pixelIndex];

        // Gamma encoding of images is used to optimize the usage of bits
        // when encoding an image, or bandwidth used to transport an image,
        // by taking advantage of the non-linear manner in which humans perceive
        // light and color. (wikipedia)

        // we use gamma 2: raising the color to the power 1/gamma (1/2)
        col = vec3(sqrtf(col[0]), sqrtf(col[1]), sqrtf(col[2]));

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

CUDA_HOSTDEV bool Renderer::traceRays(std::unique_ptr<Camera>& cam, hitable* world,
                                      std::unique_ptr<Image>& image, int sampleCount)
{
    #ifdef CUDA_ENABLED
        cudaRender(cam, world, image, sampleCount);
    #else
        // collapses the two nested fors into the same parallel for
        #pragma omp parallel for collapse(2)
        // j track rows - from top to bottom
        for (int j = 0; j < image->ny; j++)
        {
            // i tracks columns - left to right
            for (int i = 0; i < image->nx; i++)
            {
                render(i, j, cam, image, world, sampleCount);
            }
        }

        // Denoise here.
        #ifdef OIDN_ENABLED
            image->denoise();
        #endif // OIDN_ENABLED

        #pragma omp parallel for collapse(2)
        // j track rows - from top to bottom
        for (int j = 0; j < image->ny; j++)
        {
            // i tracks columns - left to right
            for (int i = 0; i < image->nx; i++)
            {
                display(i, j, image);
            }
        }
    #endif // CUDA_ENABLED
    return true;
}
