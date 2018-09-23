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

#define nx 1200
#define ny 600
#define ns 100          // sample size

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

// defocus blur = depth of field
/*
Peter Shirley's book:
The reason we defocus blur in real cameras is because they need a big hole (rather than just a
pinhole) to gather light. This would defocus everything, but if we stick a lens in the hole, there
will be a certain distance where everything is in focus. The distance to that plane where things
are in focus is controlled by the distance between the lens and the film/sensor.

For a real camera, if you need more light you make the aperture bigger, and
will get more defocus blur. For our virtual camera, we can have a perfect sensor and never
need more light, so we only have an aperture when we want defocus blur.
*/

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

void sceneExample()
{
    // for png file
    uint8_t *image = new uint8_t[nx * ny * 3];

    std::ofstream myfile ("test.ppm");
    if (myfile.is_open())
    {
        myfile << "P3\n" << nx << " " << ny << "\n255\n";
        
        hitable *world = randomScene();
        //float R = cos(M_PI/4.0f);
        //list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
        //list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
        //list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 0.3f));
        //list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5));
        //list[4] = new sphere(vec3(-1.0f, 0.0f, -1.0f), -0.45f, new dielectric(1.5));
        
        //hitable *world = new hitableList(list, 5);

        vec3 lookFrom(13.0f, 2.0f, 3.0f);
        vec3 lookAt(0.0f, 0.0f, 0.0f);
        float distToFocus = 10.0f;
        float aperture = 0.1;

        camera cam(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), aperture, distToFocus);
        
        // create source of randomness, and initialize it with non-deterministic seed
        std::random_device r;
        std::mt19937 mt(r());
        // a distribution that takes randomness and produces values in specified range
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // j track rows - from top to bottom
        for (int j = ny-1; j >= 0; j--)
        {
            // i tracks columns - left to right
            for (int i = 0; i < nx; i++)
            {
                vec3 col(0.0f, 0.0f, 0.0f);
                #pragma omp parallel for
                for (int s = 0; s < ns; s++)
                {
                    float u = float(i + dist(mt)) / float(nx); // left to right
                    float v = float(j + dist(mt)) / float(ny); // bottom to top
                    
                    ray r = cam.getRay(u,v);

                    col += color(r, world, 0);
                }
                col /= float(ns);
                
                // Gamma encoding of images is used to optimize the usage of bits 
                // when encoding an image, or bandwidth used to transport an image, 
                // by taking advantage of the non-linear manner in which humans perceive 
                // light and color. (wikipedia)
                
                // we use gamma 2: raising the color to the power 1/gamma (1/2)
                col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

                int ir = int(255.99f*col[0]);
                int ig = int(255.99f*col[1]);
                int ib = int(255.99f*col[2]);

                // PNG
                int index = (ny - 1 - j) * nx + i;
                int index3 = 3 * index;

                image[index3 + 0] = ir;
                image[index3 + 1] = ig;
                image[index3 + 2] = ib;
                myfile << ir << " " << ig << " " << ib << "\n";
            }
        }
        myfile.close();
    }
    else std::cout << "Unable to open file";

    // write png
    stbi_write_png("test.png", nx, ny, 3, image, nx * 3);
    delete[] image;
}
