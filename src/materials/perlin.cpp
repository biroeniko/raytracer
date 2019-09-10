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

CUDA_DEV static float* perlinGenerate(RandomGenerator& rng)
{
    float* p = new float[256];
    for (int i = 0; i < 256; ++i)
        p[i] = rng.get1f();

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

RandomGenerator Perlin::rng;
float* Perlin::randomFloat = perlinGenerate(Perlin::rng);
int* Perlin::permX = perlinGeneratePerm(Perlin::rng);
int* Perlin::permY = perlinGeneratePerm(Perlin::rng);
int* Perlin::permZ = perlinGeneratePerm(Perlin::rng);

