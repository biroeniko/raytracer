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

#include "util/vec3.h"
#include "util/ray.h"

inline float ffmin(float a, float b)
{
    return a < b ? a : b;
}

inline float ffmax(float a, float b)
{
    return a > b ? a : b;
}

class aabb
{
    vec3 aabbMin;
    vec3 aabbMax;

    public:
        aabb() {}
        aabb(const vec3& a, const vec3& b)
        {
            aabbMin = a;
            aabbMax = b;
        }

        vec3 min() const
        {
            return aabbMin;
        }

        vec3 max() const
        {
            return aabbMax;
        }

        bool hit(const ray& r, float tMin, float tMax) const;

};

inline bool aabb::hit(const ray& r, float tMin, float tMax) const
{
    for (int a = 0; a < 3; a++)
    {
        float t0 = ffmin(
                    (aabbMin[a] - r.origin()[a]) / r.direction()[a],
                    (aabbMax[a] - r.origin()[a]) / r.direction()[a]
                    );
        float t1 = ffmax(
                    (aabbMin[a] - r.origin()[a]) / r.direction()[a],
                    (aabbMax[a] - r.origin()[a]) / r.direction()[a]
                    );

        tMin = ffmax(t0, tMin);
        tMax = ffmin(t1, tMax);

        if (tMax <= tMin)
            return false;
    }
    return true;
}
