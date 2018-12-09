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

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "util/common.h"

class vec3 
{
    float e[3];
    
    public:
        CUDA_HOSTDEV vec3() {}
        CUDA_HOSTDEV vec3(float x, float y, float z) {e[0] = x; e[1] = y; e[2] = z;}
        CUDA_HOSTDEV float x() const {return e[0];}
        CUDA_HOSTDEV float y() const {return e[1];}
        CUDA_HOSTDEV float z() const {return e[2];}
        CUDA_HOSTDEV float r() const {return e[0];}
        CUDA_HOSTDEV float g() const {return e[1];}
        CUDA_HOSTDEV float b() const {return e[2];}

        CUDA_HOSTDEV const vec3& operator+() const;
        CUDA_HOSTDEV vec3 operator-() const;

        // Array subscript operator
        // User-defined classes that provide array-like access that allows both reading and writing typically define two overloads for operator[]: const and non-const variants:
        // like in https://en.cppreference.com/w/cpp/language/operators
        CUDA_HOSTDEV float operator[](int i) const;
        CUDA_HOSTDEV float& operator[](int i);

        CUDA_HOSTDEV vec3& operator+=(const vec3 &v2);
        CUDA_HOSTDEV vec3& operator-=(const vec3 &v2);
        CUDA_HOSTDEV vec3& operator*=(const vec3 &v2);
        CUDA_HOSTDEV vec3& operator/=(const vec3 &v2);
        CUDA_HOSTDEV vec3& operator*=(const float x);
        CUDA_HOSTDEV vec3& operator/=(const float x);

        CUDA_HOSTDEV float length() const;
        CUDA_HOSTDEV float squaredLength() const;
        CUDA_HOSTDEV void  makeUnitVector();

        // The friend declaration appears in a class body and grants a function or another class access to private and protected members of the class where the friend declaration appears.
        // the non-member function operator<< will have access to Y's private members
        friend std::istream& operator>>(std::istream &is, vec3 &t);
        friend std::ostream& operator <<(std::ostream &ps, const vec3 &t);

        CUDA_HOSTDEV friend vec3 operator+(const vec3 &v1, const vec3 &v2);
        CUDA_HOSTDEV friend vec3 operator-(const vec3 &v1, const vec3 &v2);
        CUDA_HOSTDEV friend vec3 operator*(const vec3 &v1, const vec3 &v2);
        CUDA_HOSTDEV friend vec3 operator/(const vec3 &v1, const vec3 &v2);

        CUDA_HOSTDEV friend vec3 operator*(float t, const vec3 &v);
        CUDA_HOSTDEV friend vec3 operator*(const vec3 &v, float t);
        CUDA_HOSTDEV friend vec3 operator/(const vec3 &v, float t);
        CUDA_HOSTDEV friend float dot(const vec3 &v1, const vec3 &v2);
        CUDA_HOSTDEV friend vec3 cross(const vec3 &v1, const vec3 &v2);
};

CUDA_HOSTDEV vec3 unitVector(const vec3 v);
CUDA_HOSTDEV vec3 operator+(const vec3 &v1, const vec3 &v2);
CUDA_HOSTDEV vec3 operator-(const vec3 &v1, const vec3 &v2);
CUDA_HOSTDEV vec3 operator*(const vec3 &v1, const vec3 &v2);
CUDA_HOSTDEV vec3 operator/(const vec3 &v1, const vec3 &v2);
CUDA_HOSTDEV vec3 operator*(float t, const vec3 &v);
CUDA_HOSTDEV vec3 operator*(const vec3 &v, float t);
CUDA_HOSTDEV vec3 operator/(const vec3 &v, float t);
CUDA_HOSTDEV float dot(const vec3 &v1, const vec3 &v2);
CUDA_HOSTDEV vec3 cross(const vec3 &v1, const vec3 &v2);
