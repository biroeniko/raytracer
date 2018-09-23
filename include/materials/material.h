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

/*
As in Peter Shirley's book:
- diffuse = matte materials
- diffuse objects that don't emit light take on the color of their surroundings
- BUT they modulate that with their own intrinsic colot
- light that reflects off a diffuse surface has its direction randomized
- if we send three rays into a crack between two diffuse surfaces they will each have different random behavior
- rays might be absorbed
- the darker the durface, the more likely absorption is
*/

#pragma once

struct hitable;

#include <random>
#include "util/ray.h"
#include "hitables/hitable.h"

vec3 randomInUnitSphere()
{
    std::random_device r;
    std::mt19937 mt(r());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vec3 point;
    do {
        point = 2.0f * vec3(dist(mt), dist(mt), dist(mt)) - vec3(1.0f,1.0f,1.0f);
    } while (point.squaredLength() >= 1.0f);
    return point;
}

class material
{
    public:
        virtual bool scatter(const ray& rIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const = 0;
};

// lambertian (diffuse)
// it can either scatter always and attenuate by its reflectance R
// or it can scatter with no attenuation but absorb the fraction 1-R of the rays
// or MIXED of these two strategies
class lambertian : public material 
{
    vec3 albedo; // the proportion of the incident light or radiation that is reflected by a surface
    public:
        lambertian(const vec3& a) : albedo(a) {}
        virtual bool scatter(const ray& rIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const;
};

bool lambertian::scatter(const ray& rIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const
{                    
    vec3 target = rec.point + rec.normal + randomInUnitSphere();
    scattered = ray(rec.point, target-rec.point);
    attenuation = albedo;
    return true;
}

// for smooth metals the ray won't be randomly scattered
// because v points in, we will need a minus sign before the dot product
vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2*dot(v,n)*n;
}

class metal: public material
{
    vec3 albedo;
    public:
        metal(const vec3& a) : albedo(a) {}
        virtual bool scatter(const ray& rIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const;
};

inline bool metal::scatter(const ray& rIn, const hitRecord& rec, vec3& attenuation, ray& scattered) const
{
    vec3 reflected = reflect(unitVector(rIn.direction()), rec.normal);
    scattered = ray(rec.point, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}