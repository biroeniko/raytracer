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

#include "util/vec3.h"
#include "materials/perlin.h"

class Texture
{

    public:

        CUDA_DEV virtual Vec3 value(float u, float v, const Vec3& p) const = 0;

};

class ConstantTexture : public Texture
{

    public:

        CUDA_DEV ConstantTexture() { }
        CUDA_DEV ConstantTexture(Vec3 c) : color(c) { }

        CUDA_DEV virtual Vec3 value(float u, float v, const Vec3& p) const
        {
            return color;
        }

        Vec3 color;

};

class CheckerTexture : public Texture
{

    public:

        CUDA_DEV CheckerTexture() { }
        CUDA_DEV CheckerTexture(Texture *t0, Texture *t1): even(t0), odd(t1) { }

        CUDA_DEV virtual Vec3 value(float u, float v, const Vec3& p) const
        {
            float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

        Texture *odd;
        Texture *even;

};

class NoiseTexture : public Texture
{

    public:

        CUDA_DEV NoiseTexture() {}

        CUDA_DEV virtual Vec3 value(float u, float v, const Vec3& p) const
        {
            return Vec3(1.0f,1.0f,1.0f) * noise.noise(p);
        }

        Perlin noise;
};
