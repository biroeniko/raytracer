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

CUDA_DEV static Vec3* perlinGenerate(RandomGenerator& rng);

CUDA_DEV static void permute(RandomGenerator& rng, int *p, int n);

CUDA_DEV static int* perlinGeneratePerm(RandomGenerator& rng);

CUDA_DEV float perlinInterp(Vec3 c[2][2][2], float u, float v, float w);

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
            int i = floor(p.x());
            int j = floor(p.y());
            int k = floor(p.z());
            Vec3 c[2][2][2];
            for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                    for (int dk = 0; dk < 2; dk++)
                        c[di][dj][dk] = randomVector[permX[(i+di) & 255] ^ permY[(j+dj) & 255] ^ permZ[(k+dk) & 255]];
            return perlinInterp(c, u, v, w);

        }

        CUDA_DEV float turb(const Vec3& p, int depth=7) const
        {
            float accum = 0;
            Vec3 tempP = p;
            float weight = 1.0;
            for (int i = 0; i < depth; i++)
            {
                accum += weight*noise(tempP);
                weight *= 0.5;
                tempP *= 2;
            }
            return fabs(accum);
        }

        static Vec3* randomVector;
        static int *permX;
        static int *permY;
        static int *permZ;

};


