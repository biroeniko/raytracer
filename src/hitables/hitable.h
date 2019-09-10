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

#include "util/ray.h"
#include "aabb.h"

class Material;

struct HitRecord
{

    float time;
    Vec3 point;
    Vec3 normal;
    Material* matPtr;

};

class Hitable
{

    public:
        // the hit counts if tMin < t < tMax
        // for the initial rays this is positive t
        // compute the normal if we hit something?
        // we will only need the normal of the closest thing
        // we want motion blur => time input variable
        CUDA_DEV virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;

        CUDA_DEV virtual bool boundingBox(float t0, float t1, AABB& box) const = 0;

        CUDA_DEV virtual ~Hitable() {}

};
