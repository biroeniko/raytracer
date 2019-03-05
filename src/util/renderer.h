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

#include <iostream>
#include <random>
#include <float.h>
#include <omp.h>

#include "hitables/hitableList.h"
#include "util/camera.h"
#include "util/image.h"
#include "util/randomGenerator.h"
#include "materials/material.h"
#include "hitables/sphere.h"

class Renderer
{
    bool showWindow;
    bool writeImagePPM;
    bool writeImagePNG;

    public:
        CUDA_HOSTDEV Renderer(bool showWindow, bool writeImagePPM, bool writeImagePNG) : showWindow(showWindow), writeImagePPM(writeImagePPM), writeImagePNG(writeImagePNG) {}

        CUDA_DEV vec3 color(RandomGenerator& rng, const ray& r, hitable* world, int depth)
        {
            ray curRay = r;
            vec3 curAttenuation = vec3(1.0f, 1.0f, 1.0f);
            for (int i = 0; i < 50; i++)
            {
                hitRecord rec;
                if (world->hit(curRay, 0.001f, FLT_MAX, rec))
                {
                    ray scattered;
                    vec3 attenuation;
                    if (rec.matPtr->scatter(rng, curRay, rec, attenuation, scattered))
                    {
                        curAttenuation *= attenuation;
                        curRay = scattered;
                    }
                    else
                        return vec3(0.0f, 0.0f, 0.0f);
                }
                else
                {
                    vec3 unit_direction = unitVector(curRay.direction());
                    float t = 0.5f * (unit_direction.y() + 1.0f);
                    vec3 c = (1.0f-t) * vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
                    return curAttenuation * c;
                }
            }
            return vec3(0.0f, 0.0f, 0.0f); // exceeded recursion
        }

        CUDA_HOSTDEV bool traceRays(uint32_t* windowPixels, Camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage);

        #ifdef CUDA_ENABLED
            void cudaRender(Camera* cam, hitable* world, Image* image, int sampleCount);
        #else
            CUDA_HOSTDEV void render(int i, int j, Camera* cam, Image* image, hitable* world, int sampleCount);
        #endif // CUDA_ENABLED
};
