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
#include "hitables/sphere.h"
#include "hitables/movingsphere.h"
#include "hitables/hitablelist.h"
#include "materials/material.h"
#include "materials/texture.h"
#include "util/randomgenerator.h"
#include "util/common.h"

#ifdef CUDA_ENABLED
    CUDA_GLOBAL void simpleScene(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

            *world  = new HitableList(list, 4);

            //*world = new bvhNode(list, 4, 0.0, 1.0);
        }

    }

    CUDA_GLOBAL void simpleScene2(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int count = 58;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));
            int i = 4;

            for (int a = -3; a < 3; a++)
            {
                for (int b = -4; b < 5; b++)
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

            *world = new HitableList(list, count);
        }

    }

    CUDA_GLOBAL void randomScene(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 901;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.4f, 0.2f, 0.1f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

            int i = 4;
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

            *world = new HitableList(list, n);
        }

    }

    CUDA_GLOBAL void randomScene2(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 102;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.3f, 0.0f, 0.0f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.4f, 0.5f, 0.6f), 0.0f));

            int i = 4;
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

            *world = new HitableList(list, n);
        }

    }

    CUDA_GLOBAL void randomScene3(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 68;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.3f, 0.0f, 0.0f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.4f, 0.5f, 0.6f), 0.0f));

            int i = 4;
            for (int a = -4; a < 4; a++)
            {
                for (int b = -4; b < 4; b++)
                {
                    float chooseMat = rng.get1f();
                    Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
                    if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
                    {
                        if (chooseMat < 0.3f)            // diffuse
                        {
                            list[i++] = new Sphere(center, 0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                        }
                        else if (chooseMat < 0.65f)      // metal
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

            *world = new HitableList(list, n);
        }

    }

    CUDA_GLOBAL void randomScene4(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 197;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.3f, 0.0f, 0.0f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.4f, 0.5f, 0.6f), 0.0f));

            int i = 4;
            for (int a = -7; a < 7; a++)
            {
                for (int b = -7; b < 7; b++)
                {
                    float chooseMat = rng.get1f();
                    Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
                    if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
                    {
                        if (chooseMat < 0.33f)            // diffuse
                        {
                            list[i++] = new Sphere(center, 0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                        }
                        else if (chooseMat < 0.88f)      // metal
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

            *world = new HitableList(list, n);
        }

    }

    CUDA_GLOBAL void randomSceneWithMovingSpheres(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 197;
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(new ConstantTexture(Vec3(0.5f, 0.5f, 0.5f))));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.3f, 0.0f, 0.0f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.4f, 0.5f, 0.6f), 0.0f));

            int i = 4;
            for (int a = -7; a < 7; a++)
            {
                for (int b = -7; b < 7; b++)
                {
                    float chooseMat = rng.get1f();
                    Vec3 center(a+0.9f*rng.get1f(), 0.2f, b+0.9f*rng.get1f());
                    if ((center-Vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f)
                    {
                        if (chooseMat < 0.33f)            // diffuse
                        {
                            list[i++] = new MovingSphere(center, center+Vec3(0.0f, 0.5f*rng.get1f(), 0.0f), 0.0f, 1.0f,
                                            0.2f, new Lambertian(new ConstantTexture(Vec3(rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f(), rng.get1f()*rng.get1f()))));
                        }
                        else if (chooseMat < 0.88f)      // metal
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

            *world = new HitableList(list, n);
        }

    }

    CUDA_GLOBAL void randomSceneTexture(Hitable** list, Hitable** world)
    {

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            RandomGenerator rng;

            int n = 102;
            Texture *checker = new CheckerTexture(
                new ConstantTexture(Vec3(0.9, 0.05, 0.08)),
                new ConstantTexture(Vec3(0.9, 0.9, 0.9))
            );
            list[0] = new Sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new Lambertian(checker));
            list[1] = new Sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, new Dielectric(1.5f));
            list[2] = new Sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, new Lambertian(new ConstantTexture(Vec3(0.3f, 0.0f, 0.0f))));
            list[3] = new Sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, new Metal(Vec3(0.4f, 0.5f, 0.6f), 0.0f));

            int i = 4;
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

            *world = new HitableList(list, n);
        }

    }

#endif // CUDA_ENABLED
