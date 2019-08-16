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

class movingSphere: public hitable
{
    public:
        vec3 center0, center1;
        float time0, time1;
        float radius;
        material *matPtr;

        CUDA_DEV movingSphere() {}
        CUDA_DEV movingSphere(vec3 cen0, vec3 cen1,
            float t0, float t1, float r, material *m) :
            center0(cen0), center1(cen1),
            time0(t0), time1(t1),
            radius(r), matPtr(m) {}

        CUDA_DEV bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
        CUDA_DEV vec3 center(float time) const;
        CUDA_DEV bool boundingBox(float t0, float t1, aabb& box) const override;

};

inline CUDA_DEV bool movingSphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const
{

    vec3 oc = r.origin() - center(r.time());
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
            rec.normal = (rec.point - center(r.time())) / radius;
            rec.matPtr = matPtr;
            return true;
        }
        temp = (-b + static_cast<float>(sqrt(static_cast<double>(discriminant))))/a;
        if (temp < tMax && temp > tMin)
        {
            rec.time = temp;
            rec.point = r.pointAtParameter(rec.time);
            rec.normal = (rec.point - center(r.time())) / radius;
            rec.matPtr = matPtr;
            return true;
        }
    }
    return false;
}


inline CUDA_DEV vec3 movingSphere::center(float time) const
{
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

inline CUDA_DEV bool movingSphere::boundingBox(float t0, float t1, aabb& box) const
{
        aabb box0(center(t0) - vec3(radius, radius, radius),
                  center(t0) + vec3(radius, radius, radius)
                 );
        aabb box1(center(t1) - vec3(radius, radius, radius),
                  center(t1) + vec3(radius, radius, radius)
                 );
        box = surroundingBox(box0, box1);
        return true;
}
