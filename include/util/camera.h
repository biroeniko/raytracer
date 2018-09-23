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

#include "util/ray.h"

// fov - field of view
// image is not square => fow is different horizontally and  vertically

class camera
{
    public:
        vec3 origin;
        vec3 lowerLeftCorner;
        vec3 horizontal;
        vec3 vertical;

        /*camera():   lowerLeftCorner(vec3(-2.0f, -1.0f, -1.0f)), 
                    horizontal(vec3(4.0f, 0.0f, 0.0f)),
                    vertical(vec3(0.0f, 2.0f, 0.0f)),
                    origin(vec3(0.0f, 0.0f, 0.0f)) {};
        camera(float vfov, float aspect);*/

        // vfov is top to bottom in degrees
        camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect)
        {
            vec3 u, v, w;
            float theta = vfov*M_PI/180;
            float halfHeight = tan(theta/2.0f);
            float halfWidth = aspect * halfHeight;
            
            origin = lookFrom;
            w = unitVector(lookFrom - lookAt);
            u = unitVector(cross(vup, w));
            v = cross(w, u);
            
            lowerLeftCorner = vec3(-halfWidth, -halfHeight, -1.0);
            lowerLeftCorner = origin - halfWidth*u - halfHeight*v - w;
            horizontal = 2.0f*halfWidth*u;
            vertical = 2.0f*halfHeight*v;
        } 
        ray getRay(float u, float v) {return ray(origin, lowerLeftCorner + u*horizontal + v*vertical - origin);}
};
/*
camera::camera(float vfov, float aspect)
{
    float theta = vfov*M_PI/180.0f;
    float halfHeight = tan(theta/2.0f);
    float halfWidth = aspect * halfHeight;
    lowerLeftCorner = vec3(-halfWidth, -halfHeight, -1.0);
    horizontal = vec3(2*halfWidth, 0.0, 0.0);
    vertical = vec3(0.0, 2*halfHeight, 0.0);
    origin = vec3(0.0, 0.0, 0.0);
}*/
