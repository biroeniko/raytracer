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

#ifdef CUDA_ENABLED
    CUDA_GLOBAL void render(Camera* cam, vec3* pixels, uint32_t* windowPixels, uint8_t* fileOutputImage, bool writeImagePNG, bool showWindow, int nx, int ny, int sampleCount)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= nx) || (j >= ny))
            return;

        RandomGenerator rng(sampleCount, i*nx + j);
        float u = float(i + rng.get1f()) / float(nx); // left to right
        float v = float(j + rng.get1f()) / float(ny); // bottom to top

        int pixelIndex = j*nx + i;

        //ray r = cam->getRay(rng, u, v);

        vec3 col(0.0f,1.0f,0.0f);

        pixels[pixelIndex] = col;

        int ir = int(255.99f*col[0]);
        int ig = int(255.99f*col[1]);
        int ib = int(255.99f*col[2]);

        if (writeImagePNG)
        {
            // PNG
            int index = (ny - 1 - j) * nx + i;
            int index3 = 3 * index;

            fileOutputImage[index3 + 0] = ir;
            fileOutputImage[index3 + 1] = ig;
            fileOutputImage[index3 + 2] = ib;
        }

        if (showWindow)
            windowPixels[(ny-j-1)*nx + i] = (ir << 16) | (ig << 8) | (ib);


        /*
        RandomGenerator rng(sampleCount, i*image->rows + j);
        float u = float(i + rng.get1f()) / float(image->rows); // left to right
        float v = float(j + rng.get1f()) / float(image->columns); // bottom to top

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

        if (writeImagePNG)
        {
            // PNG
            int index = (image->columns - 1 - j) * image->rows + i;
            int index3 = 3 * index;

            fileOutputImage[index3 + 0] = ir;
            fileOutputImage[index3 + 1] = ig;
            fileOutputImage[index3 + 2] = ib;
        }

        if (showWindow)
            windowPixels[(image->columns-j-1)*image->rows + i] = (ir << 16) | (ig << 8) | (ib);

        */
    }
#endif // CUDA_ENABLED

#ifdef CUDA_ENABLED
    void Renderer::cudaRender(uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage)
    {
        dim3 blocks(image->nx/image->tx+1, image->ny/image->ty+1);
        dim3 threads(image->tx, image->ty);

        render<<<blocks, threads>>>(cam, image->pixels, image->windowPixels, image->fileOutputImage, writeImagePNG, showWindow, image->nx, image->ny, sampleCount);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
#endif // CUDA_ENABLED
