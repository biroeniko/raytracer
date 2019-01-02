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

#ifdef CUDA_ENABLED

#else
    CUDA_HOSTDEV void Renderer::render(int i, int j, uint32_t* windowPixels, Camera* cam, hitable** world, Image* image, int sampleCount, uint8_t *fileOutputImage)
    {
        RandomGenerator rng(sampleCount, i*image->nx + j);
        float u = float(i + rng.get1f()) / float(image->nx); // left to right
        float v = float(j + rng.get1f()) / float(image->ny); // bottom to top

        ray r = cam->getRay(rng, u,v);

        image->pixels[i][j] += color(rng, r, world, 0);

        vec3 col = image->pixels[i][j] / sampleCount;

        // Gamma encoding of images is used to optimize the usage of bits
        // when encoding an image, or bandwidth used to transport an image,
        // by taking advantage of the non-linear manner in which humans perceive
        // light and color. (wikipedia)

        // we use gamma 2: raising the color to the power 1/gamma (1/2)
        col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

        int ir = int(255.99f*col[0]);
        int ig = int(255.99f*col[1]);
        int ib = int(255.99f*col[2]);

        if (writeImagePNG || writeImagePPM)
        {
            // PNG
            int index = (image->ny - 1 - j) * image->nx + i;
            int index3 = 3 * index;

            fileOutputImage[index3 + 0] = ir;
            fileOutputImage[index3 + 1] = ig;
            fileOutputImage[index3 + 2] = ib;
        }

        if (showWindow)
            windowPixels[(image->ny-j-1)*image->nx + i] = (ir << 16) | (ig << 8) | (ib);
    }
#endif // CUDA_ENABLED

CUDA_HOSTDEV bool Renderer::traceRays(uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage)
{
    #ifdef CUDA_ENABLED
        cudaRender(windowPixels, cam, world, image, sampleCount, fileOutputImage);
    #else
        // collapses the two nested fors into the same parallel for
        #pragma omp parallel for collapse(2)
        // j track rows - from top to bottom
        for (int j = 0; j < image->ny; j++)
        {
            // i tracks columns - left to right
            for (int i = 0; i < image->nx; i++)
            {

                render(i, j, windowPixels, cam, world, image, sampleCount, fileOutputImage);
            }
        }

    #endif // CUDA_ENABLED
    return true;
}
