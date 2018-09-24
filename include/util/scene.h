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

#include <iostream>
#include <fstream>
#include <float.h>
#include <random>
#include <SDL2/SDL.h>

#ifndef STB_IMAGE_IMPLEMENTATION 
  #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
#endif /* STB_IMAGE_IMPLEMENTATION */

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION 
  #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
#endif /* STB_IMAGE_WRITE_IMPLEMENTATION */

#include "hitables/sphere.h"
#include "hitables/hitableList.h"
#include "util/camera.h"
#include "materials/material.h"

#pragma once

vec3 color(const ray& r, hitable *world, int depth)
{
    hitRecord rec;
    if (world->hit(r, 0.001f, FLT_MAX, rec))        // get rid of shadow acne problem
    {
        ray scattered;
        vec3 attenuation;
        if (depth < 50 && rec.matPtr->scatter(r, rec, attenuation, scattered))
            return attenuation*color(scattered, world, depth+1);
        else
            return vec3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        // background
        vec3 unitDirection = unitVector(r.direction());
        float t = 0.5f*(unitDirection.y() + 1.0f);
        return (1.0f-t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
    }
}


hitable* simpleScene()
{
    hitable** list = new hitable*[4];
    list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
    list[1] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    list[2] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
    list[3] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    return new hitableList(list, 4);
}

hitable* randomScene()
{
    int n = 1000;
    hitable** list = new hitable*[n+1];
    list[0] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));
    int i = 1;
    for (int a = -15; a < 15; a++)
    {
        for (int b = -15; b < 15; b++)
        {
            float chooseMat = dist(mt);
            vec3 center(a+0.9f*dist(mt), 0.2f, b+0.9f*dist(mt));
            if ((center-vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.2)            // diffuse
                {
                    list[i++] = new sphere(center, 0.2f, new lambertian(vec3(dist(mt)*dist(mt), dist(mt)*dist(mt), dist(mt)*dist(mt))));
                }
                else if (chooseMat < 0.35)      // metal 
                {
                    list[i++] = new sphere(center, 0.2f, new metal(vec3(0.5*(1+dist(mt)), 0.5*(1+dist(mt)), 0.5*(1+dist(mt))))); 
                }
                else                            // glass
                {
                    list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
    list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    return new hitableList(list, i);
}