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
#include "util/util.h"

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// fov - field of view
// image is not square => fow is different horizontally and vertically

class Camera
{
    public:
        vec3 origin;
        vec3 lowerLeftCorner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lensRadius;

        vec3 lookFrom;
        vec3 lookAt;

        vec3 vup;
        float vfov;
        float aspect;
        float aperture;
        float focusDist;

        float halfWidth;
	    float halfHeight;

        Camera():   lowerLeftCorner(vec3(-2.0f, -1.0f, -1.0f)), 
                    horizontal(vec3(4.0f, 0.0f, 0.0f)),
                    vertical(vec3(0.0f, 2.0f, 0.0f)),
                    origin(vec3(0.0f, 0.0f, 0.0f)) {};
        
        // vfov is top to bottom in degrees
        Camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect)
        {
            float theta = vfov*M_PI/180.0f;
            this->halfHeight = tan(theta/2.0f);
            this->halfWidth = aspect * halfHeight;
            
            this->origin = lookFrom;
            this->w = unitVector(lookFrom - lookAt);
            this->u = unitVector(cross(vup, w));
            this->v = cross(w, u);
            
            this->lowerLeftCorner = origin - halfWidth*u - halfHeight*v - w;
            this->horizontal = 2.0f*halfWidth*u;
            this->vertical = 2.0f*halfHeight*v;
            
            this->lookFrom = lookFrom;
            this->lookAt = lookAt;

            this->vup = unitVector(vup);
            this->vfov = vfov;
            this->aspect = aspect;
        } 

        // another constructor
        Camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect, float focusDist, float aperture = 0.0f) :
        Camera(lookFrom, lookAt, vup, vfov, aspect)
        {
            this->lensRadius = aperture/2.0f;
            this->aperture = aperture;
            this->focusDist = focusDist;
        } 

        void update() 
        {
            float theta = vfov*M_PI/180.0f;
            this->halfHeight = tan(theta/2.0f);
            this->halfWidth = aspect * halfHeight;

            this->origin = lookFrom;
            this->w = unitVector(lookFrom - lookAt);
            this->u = unitVector(cross(vup, w));
            this->v = cross(w, u);
            
            this->lowerLeftCorner = origin - halfWidth*focusDist*u - halfHeight*focusDist*v - focusDist*w;
            this->horizontal = 2.0f*halfWidth*focusDist*u;
            this->vertical = 2.0f*halfHeight*focusDist*v;
	    }

        // Spherical coordinate system implementation - rotate the lookFrom location by theta polar angle and phi azimuth angle - keeping the distance 
        void rotate(float theta, float phi) 
        {
            float radialDistance = (lookFrom - lookAt).length();
            this->lookFrom = vec3(
                radialDistance*sinf(theta)*sinf(phi),
                radialDistance*cosf(theta),
                radialDistance*sinf(theta)*cosf(phi)) + lookAt;
            update();
	    }

        void zoom(float zoomScale) 
        {
            this->vfov += zoomScale;
            // min(max())
            this->vfov = clamp<float>(this->vfov, 0.0f, 180.0f);
            update();
	    }

        void translate(CameraMovement direction, float stepScale) 
        {
            if (direction == FORWARD)
            {
                lookFrom += this->w * stepScale;
                lookAt += this->w * stepScale;;
            }
            if (direction == BACKWARD)
            {
                lookFrom -= this->w * stepScale;
                lookAt -= this->w * stepScale;
            }
            if (direction == LEFT)
            {
                lookFrom -= this->u * stepScale;
                lookAt -= this->u * stepScale;
            }
            if (direction == RIGHT)
            {
                lookFrom += this->u * stepScale;
                lookAt += this->u * stepScale;
            }
            update();
	    }

        ray getRay(float s, float t) 
        {
            vec3 rd = lensRadius*randomInUnitSphere();
            vec3 offset = u * rd.x() + v * rd.y();
            return ray(origin + offset, lowerLeftCorner + s*horizontal + t*vertical - origin - offset);
        }
};
