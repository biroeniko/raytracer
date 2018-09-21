#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>

class vec3 {
    float e[3];
public:
    vec3() {}
    vec3(float x, float y, float z) {e[0] = x; e[1] = y; e[2] = z;}
    float x() const {return e[0];}
    float y() const {return e[1];}
    float z() const {return e[2];}
    float r() const {return e[0];}
    float g() const {return e[1];}
    float b() const {return e[2];}

    const vec3& operator+() const;
    vec3 operator-() const;

    // Array subscript operator
    // User-defined classes that provide array-like access that allows both reading and writing typically define two overloads for operator[]: const and non-const variants:
    // like in https://en.cppreference.com/w/cpp/language/operators
    inline float operator[](int i) const {return e[i];}
    inline float& operator[](int i) {return e[i];}

    vec3& operator+=(const vec3 &v2);
    vec3& operator-=(const vec3 &v2);
    vec3& operator*=(const vec3 &v2);
    vec3& operator/=(const vec3 &v2);
    vec3& operator*=(const float x);
    vec3& operator/=(const float x);

    float length() const;
    float squaredLength() const;
    void  makeUnitVector();

    // The friend declaration appears in a class body and grants a function or another class access to private and protected members of the class where the friend declaration appears.
    // the non-member function operator<< will have access to Y's private members
    friend std::istream& operator>>(std::istream &is, vec3 &t);
    friend std::ostream& operator <<(std::ostream &ps, const vec3 &t);

    friend vec3 operator+(const vec3 &v1, const vec3 &v2);
    friend vec3 operator-(const vec3 &v1, const vec3 &v2);
    friend vec3 operator*(const vec3 &v1, const vec3 &v2);
    friend vec3 operator/(const vec3 &v1, const vec3 &v2);

    friend vec3 operator*(float t, const vec3 &v);
    friend vec3 operator*(const vec3 &v, float t);
    friend vec3 operator/(const vec3 &v, float t);
    friend float dot(const vec3 &v1, const vec3 &v2);
    friend vec3 cross(const vec3 &v1, const vec3 &v2);
};
