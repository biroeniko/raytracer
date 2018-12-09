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

#include <random>

#include "pcg_random.hpp"
#include "util/vec3.h"

// PCG32 random number generator
// This code is based on the O'Neill implementation seen here:
// (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

class RandomGenerator
{
    uint64_t state;             // RNG state
    uint64_t inc;               // sequence

    static constexpr uint64_t defaultSeed = 0xcafef00dd1eea5e5ULL;
    static constexpr uint64_t defaultSeq  = 1442695040888963407ULL >> 1;

public:
    explicit RandomGenerator(uint64_t seed = defaultSeed, uint64_t seq = defaultSeq)
    {
        reset(seed, seq);
    }

    void reset(uint64_t seed = defaultSeed, uint64_t seq = defaultSeq)
    {
        inc = (seq << 1) | 1;
        state = seed + inc;
        next();
    }

    void next()
    {
        state = state * 6364136223846793005ULL + inc;
    }

    uint64_t getSeq() const
    {
        return inc >> 1;
    }

    uint32_t get1ui()
    {
        const uint64_t oldState = state;
        next();
        const uint32_t xorShifted = ((oldState >> 18u) ^ oldState) >> 27u;
        const uint32_t rot = oldState >> 59u;
        return (xorShifted >> rot) | (xorShifted << ((-rot) & 31u));
    }

    float toFloatUnorm(int x)
    {
        return float(uint32_t(x)) * 0x1.0p-32f;
    }

    float get1f()
    {
       return toFloatUnorm(get1ui());
    }

    vec3 randomInUnitSphere();
};

