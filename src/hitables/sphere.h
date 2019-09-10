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

#include "hitables/hitable.h"

class Sphere: public Hitable
{

    public:

        Vec3 center;
        float radius;
        Material *matPtr;

        CUDA_DEV Sphere() {}
        CUDA_DEV Sphere(Vec3 cen, float r, Material *m) : center(cen), radius(r), matPtr(m) {}

        CUDA_DEV bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
        CUDA_DEV bool boundingBox(float t0, float t1, AABB& box) const override;

};

inline CUDA_DEV bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{

    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;

    if (discriminant > 0)
    {
        float temp = (-b - static_cast<float>(sqrt(static_cast<double>(discriminant))))/a;
        if (temp < tMax && temp > tMin)
        {
            rec.time = temp;
            rec.point = r.pointAtParameter(rec.time);
            rec.normal = (rec.point - center) / radius;
            rec.matPtr = matPtr;
            return true;
        }
        temp = (-b + static_cast<float>(sqrt(static_cast<double>(discriminant))))/a;
        if (temp < tMax && temp > tMin)
        {
            rec.time = temp;
            rec.point = r.pointAtParameter(rec.time);
            rec.normal = (rec.point - center) / radius;
            rec.matPtr = matPtr;
            return true;
        }
    }

    return false;

}

inline CUDA_DEV bool Sphere::boundingBox(float t0, float t1, AABB& box) const
{

    box = AABB(center - Vec3(radius, radius, radius),
               center + Vec3(radius, radius, radius)
              );

    return true;

}
