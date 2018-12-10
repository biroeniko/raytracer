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

#include "util/camera.h"

// vfov is top to bottom in degrees
CUDA_HOSTDEV Camera::Camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect)
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

CUDA_HOSTDEV Camera::Camera(vec3 lookFrom, vec3 lookAt, vec3 vup, float vfov, float aspect, float focusDist, float aperture) :
Camera(lookFrom, lookAt, vup, vfov, aspect)
{
    this->lensRadius = aperture/2.0f;
    this->aperture = aperture;
    this->focusDist = focusDist;
} 

CUDA_HOSTDEV void Camera::update()
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

CUDA_HOSTDEV void Camera::rotate(float theta, float phi)
{
    float radialDistance = (lookFrom - lookAt).length();
    this->lookFrom = vec3(
        radialDistance*sinf(theta)*sinf(phi),
        radialDistance*cosf(theta),
        radialDistance*sinf(theta)*cosf(phi)) + lookAt;
    update();
}

CUDA_HOSTDEV void Camera::zoom(float zoomScale)
{
    this->vfov += zoomScale;
    // min(max())
    this->vfov = clamp<float>(this->vfov, 0.0f, 180.0f);
    update();
}

CUDA_HOSTDEV void Camera::translate(CameraMovement direction, float stepScale)
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

CUDA_HOSTDEV ray Camera::getRay(RandomGenerator& rng, float s, float t)
{
    vec3 rd = lensRadius*rng.randomInUnitSphere();
    vec3 offset = u * rd.x() + v * rd.y();
    return ray(origin + offset, lowerLeftCorner + s*horizontal + t*vertical - origin - offset);
}
