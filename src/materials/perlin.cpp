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
#include "util/randomgenerator.h"
#include "materials/perlin.h"

CUDA_DEV static Vec3* perlinGenerate(RandomGenerator& rng)
{
    Vec3 *p = new Vec3[256];
    for (int i = 0; i < 256; ++i)
        p[i] = unitVector(Vec3(2*rng.get1f() - 1, 2*rng.get1f() - 1, 2*rng.get1f() - 1));
    return p;
}

CUDA_DEV void permute(RandomGenerator& rng, int *p, int n)
{
    for (int i = n-1; i > 0; i--)
    {
        int target = int(rng.get1f()*(i+1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
    return;
}

CUDA_DEV static int* perlinGeneratePerm(RandomGenerator& rng)
{
    int * p = new int[256];
    for (int i = 0; i < 256; i++)
        p[i] = i;
    permute(rng, p, 256);
    return p;
}

CUDA_DEV float perlinInterp(Vec3 c[2][2][2], float u, float v, float w)
{
    float uu = u*u*(3-2*u);
    float vv = v*v*(3-2*v);
    float ww = w*w*(3-2*w);
    float accum = 0;
    for (int i=0; i < 2; i++)
        for (int j=0; j < 2; j++)
            for (int k=0; k < 2; k++)
            {
                Vec3 weightV(u-i, v-j, w-k);
                accum += (i*uu + (1-i)*(1-uu))*
                    (j*vv + (1-j)*(1-vv))*
                    (k*ww + (1-k)*(1-ww))*dot(c[i][j][k], weightV);
            }
    return accum;
}

RandomGenerator Perlin::rng;
Vec3* Perlin::randomVector = perlinGenerate(Perlin::rng);
int* Perlin::permX = perlinGeneratePerm(Perlin::rng);
int* Perlin::permY = perlinGeneratePerm(Perlin::rng);
int* Perlin::permZ = perlinGeneratePerm(Perlin::rng);

