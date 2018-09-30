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

void glassExample()
{
    int nx = 1200;
    int ny = 600;
    int ns = 100;       // sample size

    // for png file
    uint8_t *image = new uint8_t[nx * ny * 3];

    std::ofstream myfile ("test.ppm");
    if (myfile.is_open())
    {
        myfile << "P3\n" << nx << " " << ny << "\n255\n";
        
        hitable *list[5];
        float R = cos(M_PI/4.0f);
        list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
        list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
        list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 0.3f));
        list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5));
        list[4] = new sphere(vec3(-1.0f, 0.0f, -1.0f), -0.45f, new dielectric(1.5));
        
        hitable *world = new hitableList(list, 5);

        camera cam(vec3(-2.0f, 2.0f, 1.0f), vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, 1.0f, 0.0f), 90.0f, float(nx)/float(ny));

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
}
