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

// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "util/vec3.h"

class RandomGenerator
{
    unsigned int s;

public:

    CUDA_DEV explicit RandomGenerator(int sampleId = 1, int pixelId = 1)
    {
        unsigned int hash = 0;
        hash = MurmurHash3_mix(hash, pixelId);
        hash = MurmurHash3_mix(hash, sampleId);
        hash = MurmurHash3_finalize(hash);

        s = hash;
    }

    CUDA_DEV uint32_t get1ui()
    {
        s = LCG_next(s);
        return s;
    }

    CUDA_DEV float toFloatUnorm(uint32_t x)
    {
        return float(uint32_t(x)) * 0x1.0p-32f;
    }

    CUDA_DEV float get1f()
    {
       return toFloatUnorm(get1ui());
    }

    CUDA_DEV vec3 randomInUnitSphere()
    {
        vec3 point;
        do {
            point = 2.0f * vec3(get1f(), get1f(), get1f()) - vec3(1.0f,1.0f,1.0f);
        } while (point.squaredLength() >= 1.0f);
        return point;
    }

private:

    CUDA_DEV unsigned int MurmurHash3_mix(unsigned int hash, unsigned int k)
    {
        const unsigned int c1 = 0xcc9e2d51;
        const unsigned int c2 = 0x1b873593;
        const unsigned int r1 = 15;
        const unsigned int r2 = 13;
        const unsigned int m = 5;
        const unsigned int n = 0xe6546b64;

        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;

        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;

        return hash;
    }

    CUDA_DEV unsigned int MurmurHash3_finalize(unsigned int hash)
    {
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;

        return hash;
    }

    CUDA_DEV unsigned int LCG_next(unsigned int value)
    {
        const unsigned int m = 1664525;
        const unsigned int n = 1013904223;

        return value * m + n;
    }

};
