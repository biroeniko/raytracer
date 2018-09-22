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
#include "vec3.h"

// unary operators
const vec3& vec3::operator+() const
{
    return *this;
}   

vec3 vec3::operator-() const
{
    return vec3(-e[0], -e[1], -e[2]);
}

float vec3::operator[](int i) const
{
    return (*this).e[i];
}

float& vec3::operator[](int i)
{
    return (*this).e[i];
}

// binary operators
vec3& vec3::operator+=(const vec3 &v2)
{
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    return *this;
}

vec3& vec3::operator-=(const vec3 &v2)
{
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    e[2] -= v2.e[2];
    return *this;
}

vec3& vec3::operator*=(const vec3 &v2)
{
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    e[2] *= v2.e[2];
    return *this;
}

vec3& vec3::operator/=(const vec3 &v2)
{
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    e[2] /= v2.e[2];
    return *this;
}

vec3& vec3::operator*=(const float x)
{
    e[0] *= x;
    e[1] *= x;
    e[2] *= x;
    return *this;
}

vec3& vec3::operator/=(const float x)
{
    float k = 1.0 / x;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

float vec3::length() const
{
    return sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
}

float vec3::squaredLength() const
{
    return e[0]*e[0]+e[1]*e[1]+e[2]*e[2];
}

void vec3::makeUnitVector()
{
    float k = 1.0 / (sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]));
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

vec3 unitVector(const vec3 v)
{
    return v / v.length();
}

// friend functions
// if it would be inline => should be in header
std::istream& operator>>(std::istream &is, vec3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

std::ostream& operator <<(std::ostream &os, const vec3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;    
}

vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]); 
}

vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]); 
}

vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]); 
}

vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]); 
}

vec3 operator*(float t, const vec3 &v)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

vec3 operator*(const vec3 &v, float t)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

vec3 operator/(const vec3 &v, float t)
{
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}

float dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3((v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

