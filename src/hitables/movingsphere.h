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

class MovingSphere: public Hitable
{

    public:
        Vec3 center0, center1;
        float time0, time1;
        float radius;
        Material *matPtr;

        CUDA_DEV MovingSphere() {}
        CUDA_DEV MovingSphere(Vec3 cen0, Vec3 cen1,
            float t0, float t1, float r, Material *m) :
            center0(cen0), center1(cen1),
            time0(t0), time1(t1),
            radius(r), matPtr(m) {}

        CUDA_DEV bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
        CUDA_DEV Vec3 center(float time) const;
        CUDA_DEV bool boundingBox(float t0, float t1, AABB& box) const override;

};

inline CUDA_DEV bool MovingSphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{

    Vec3 oc = r.origin() - center(r.time());
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


inline CUDA_DEV Vec3 MovingSphere::center(float time) const
{
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

inline CUDA_DEV bool MovingSphere::boundingBox(float t0, float t1, AABB& box) const
{

        AABB box0(center(t0) - Vec3(radius, radius, radius),
                  center(t0) + Vec3(radius, radius, radius)
                 );
        AABB box1(center(t1) - Vec3(radius, radius, radius),
                  center(t1) + Vec3(radius, radius, radius)
                 );
        box = surroundingBox(box0, box1);

        return true;

}
