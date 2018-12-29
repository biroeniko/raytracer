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
#include <random>
#include <omp.h>

#include "hitables/hitableList.h"
#include "util/camera.h"
#include "util/scene.h"
#include "util/image.h"
#include "util/randomGenerator.h"

class Renderer
{
    bool showWindow;
    bool writeImagePPM;
    bool writeImagePNG;

    public:
        CUDA_HOSTDEV Renderer(bool showWindow, bool writeImagePPM, bool writeImagePNG) : showWindow(showWindow), writeImagePPM(writeImagePPM), writeImagePNG(writeImagePNG) {};

        CUDA_DEV vec3 color(RandomGenerator& rng, const ray& r, hitable** world, int depth)
        {
            ray cur_ray = r;
             vec3 cur_attenuation = vec3(1.0,1.0,1.0);
             for(int i = 0; i < 50; i++) {
                 hitRecord rec;
                 if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                     ray scattered;
                     vec3 attenuation;
                     //if(rec.matPtr->scatter(rng, cur_ray, rec, attenuation, scattered)) {
                     //    cur_attenuation *= attenuation;
                     //    cur_ray = scattered;
                     //}
                     //else {
                         return vec3(0.0,0.0,0.0);
                     //}
                 }
                 else {
                     vec3 unit_direction = unitVector(cur_ray.direction());
                     float t = 0.5f*(unit_direction.y() + 1.0f);
                     vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
                     return cur_attenuation * c;
                 }
             }
             return vec3(0.0,0.0,0.0); // exceeded recursion

            /*
                ray cur_ray = r;
                vec3 cur_attenuation = vec3(1.0,1.0,1.0);


                hitRecord rec;
                if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
                    ray scattered;
                    vec3 attenuation;
                    if(rec.matPtr->scatter(rng, cur_ray, rec, attenuation, scattered)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                    }
                    else {
                        return vec3(1.0,0.0,0.0);
                    }
                }
                else {
                    return vec3(1.0,1.0,0.0);
                }

                return vec3(1.0,1.0,1.0);
*/
                /*
                hitRecord rec;
                (*world)->hit(r, 0.001f, FLT_MAX, rec);
                (*world)->test();
                //return vec3(1,1,0);
                */
/*
                ray cur_ray = r;
                vec3 cur_attenuation = vec3(1.0,1.0,1.0);
            for(int i = 0; i < 5; i++)
            {
                    hitRecord rec;
                    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                        ray scattered;
                        vec3 attenuation;
                        if(rec.matPtr->scatter(rng, cur_ray, rec, attenuation, scattered)) {
                            cur_attenuation *= attenuation;
                            cur_ray = scattered;
                        }
                        else {
                            return vec3(0.0,0.0,0.0);
                        }
                    }
                    else {
                        vec3 unit_direction = unitVector(cur_ray.direction());
                        float t = 0.5f*(unit_direction.y() + 1.0f);
                        vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
                        return cur_attenuation * c;
                    }
                }
             return vec3(0.0,0.0,0.0); // exceeded recursion
*/
/*
            hitRecord rec;
            if ((*world)->hit(r, 0.001f, FLT_MAX, rec))        // get rid of shadow acne problem
            {
                ray scattered;
                vec3 attenuation;
                //if (depth < 50 && rec.matPtr->scatter(rng, r, rec, attenuation, scattered))
                    //return attenuation*color(rng, scattered, world, depth+1);
                //else
                    return vec3(1.0f, 1.0f, 0.0f);
            }
            else
            {
                // background
                vec3 unitDirection = unitVector(r.direction());
                float t = 0.5f*(unitDirection.y() + 1.0f);
                return (1.0f-t)*vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
            }
            */

        }

        CUDA_HOSTDEV bool traceRays(uint32_t* windowPixels, Camera* cam, hitable** world, Image* image, int sampleCount, uint8_t *fileOutputImage);

        #ifdef CUDA_ENABLED
            void cudaRender(uint32_t* windowPixels, Camera* cam, hitable** world, Image* image, int sampleCount, uint8_t *fileOutputImage);
        #else
            CUDA_HOSTDEV void render(int i, int j, uint32_t* windowPixels, Camera* cam, hitable** world, Image* image, int sampleCount, uint8_t *fileOutputImage);
        #endif // CUDA_ENABLED
};
