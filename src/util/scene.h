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

#include <float.h>

#include "hitables/bvh.h"
#include "hitables/hitablelist.h"
#include "hitables/sphere.h"
#include "materials/material.h"
#include "materials/texture.h"
#include "util/randomgenerator.h"
#include "util/common.h"

CUDA_HOSTDEV inline Hitable* simpleScene()
{

    Hitable** list = new Hitable*[4];
    list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
    list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
    list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
    list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

    //return new hitableList(list, 4);
    return new BVHNode(list, 4, 0.0, 1.0);

}

CUDA_HOSTDEV inline Hitable* simpleScene2()
{

    RandomGenerator rng;

    int n = 20;
    Hitable** list = new Hitable*[n];
    list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
    list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
    list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
    list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));
    int i = 4;

    for (int a = -2; a < 2; a++)
    {
        for (int b = -2; b < 2; b++)
        {
            float chooseMat = rng.get1f();
            Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
            if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.5f)            // diffuse
                {
                    list[i++] = new Sphere(center, 0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                }
                else if (chooseMat < 0.75f)      // metal
                {
                    list[i++] = new Sphere(center, 0.2f, new Metal(Vec3(0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()))));
                }
                else                            // glass
                {
                    list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
                }
            }
        }
    }

    //return new hitableList(list, count);
    return new BVHNode(list, i, 0.0, 1.0);

}

inline Hitable* randomScene()
{

    RandomGenerator rng;

    int n = 1000;
    Hitable** list = new Hitable*[n];
    list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
    int i = 1;
    for (int a = -15; a < 15; a++)
    {
        for (int b = -15; b < 15; b++)
        {
            float chooseMat = rng.get1f();
            Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
            if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.5f)            // diffuse
                {
                    list[i++] = new Sphere(center, 0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                }
                else if (chooseMat < 0.75f)      // metal
                {
                    list[i++] = new Sphere(center, 0.2f, new Metal(Vec3(0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()))));
                }
                else                            // glass
                {
                    list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
                }
            }
        }
    }

    list[i++] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
    list[i++] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
    list[i++] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

    //return new hitableList(list, i);
    return new BVHNode(list, i, 0.0, 1.0);

}


inline Hitable* randomSceneTexture()
{

    RandomGenerator rng;

    int n = 104;
    Hitable** list = new Hitable*[n];
    Texture *checker = new CheckerTexture(
        new ConstantTexture(Vec3(0.9, 0.05, 0.08)),
        new ConstantTexture(Vec3(0.9, 0.9, 0.9))
    );
    list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(checker));
    int i = 1;
    for (int a = -5; a < 5; a++)
    {
        for (int b = -5; b < 5; b++)
        {
            float chooseMat = rng.get1f();
            Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
            if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
            {
                if (chooseMat < 0.5f)            // diffuse
                {
                    list[i++] = new Sphere(center, 0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                }
                else if (chooseMat < 0.75f)      // metal
                {
                    list[i++] = new Sphere(center, 0.2f, new Metal(Vec3(0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()), 0.5f*(1.0f+rng.get1f()))));
                }
                else                            // glass
                {
                    list[i++] = new Sphere(center, 0.2f, new Dielectric(1.5f));
                }
            }
        }
    }

    list[i++] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
    list[i++] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
    list[i++] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

    //return new hitableList(list, i);
    return new BVHNode(list, i, 0.0, 1.0);

}

