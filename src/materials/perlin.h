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

#include "util/randomgenerator.h"
#include "util/vec3.h"

CUDA_DEV static float* perlinGenerate(RandomGenerator& rng);

CUDA_DEV static void permute(RandomGenerator& rng, int *p, int n);

CUDA_DEV static int* perlinGeneratePerm(RandomGenerator& rng);

#include <iostream>

class Perlin
{

        static RandomGenerator rng;

    public:

        CUDA_DEV Perlin()
        {


        }

        CUDA_DEV float noise(const Vec3& p) const
        {

            float u = p.x() - floor(p.x());
            float v = p.y() - floor(p.y());
            float w = p.z() - floor(p.z());
            int i = int(4*p.x()) & 255;
            int j = int(4*p.y()) & 255;
            int k = int(4*p.z()) & 255;

            return randomFloat[permX[i] ^ permY[j] ^ permZ[k]];
        }

        static float* randomFloat;
        static int *permX;
        static int *permY;
        static int *permZ;

};
