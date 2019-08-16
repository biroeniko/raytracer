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

class aabb
{
    vec3 aabbMin;
    vec3 aabbMax;

    public:
        CUDA_DEV aabb() {}
        CUDA_DEV aabb(const vec3& a, const vec3& b)
        {
            aabbMin = a;
            aabbMax = b;
        }

        CUDA_DEV vec3 min() const
        {
            return aabbMin;
        }

        CUDA_DEV vec3 max() const
        {
            return aabbMax;
        }

        CUDA_DEV bool hit(const ray& r, float tMin, float tMax) const;

};

inline CUDA_DEV bool aabb::hit(const ray& r, float tMin, float tMax) const
{
    for (int a = 0; a < 3; a++)
    {
        float invD = 1.0f / r.direction()[a];
        float t0 = (min()[a] - r.origin()[a]) * invD;
        float t1 = (max()[a] - r.origin()[a]) * invD;

        if (invD < 0.0f)
            std::swap(t0, t1);

        tMin = t0 > tMin ? t0 : tMin;
        tMax = t1 < tMax ? t1 : tMax;

        if (tMax <= tMin)
            return false;
    }
    return true;
}

inline CUDA_DEV aabb surroundingBox(aabb box0, aabb box1)
{
    vec3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z())
              );
    vec3 big(fmax(box0.max().x(), box1.max().x()),
             fmax(box0.max().y(), box1.max().y()),
             fmax(box0.max().z(), box1.max().z())
            );
    return aabb(small,big);
}
