#include "vec3.h"

// unary operators
inline const vec3& vec3::operator+() const
{
    return *this;
}   

inline vec3 vec3::operator-() const
{
    return vec3(-e[0], -e[1], -e[2]);
}


// binary operators
inline vec3& vec3::operator+=(const vec3 &v2)
{
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    return *this;
}

inline vec3& vec3::operator-=(const vec3 &v2)
{
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    e[2] -= v2.e[2];
    return *this;
}

inline vec3& vec3::operator*=(const vec3 &v2)
{
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    e[2] *= v2.e[2];
    return *this;
}

inline vec3& vec3::operator/=(const vec3 &v2)
{
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    e[2] /= v2.e[2];
    return *this;
}

inline vec3& vec3::operator*=(const float x)
{
    e[0] *= x;
    e[1] *= x;
    e[2] *= x;
    return *this;
}

inline vec3& vec3::operator/=(const float x)
{
    float k = 1.0 / x;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

inline float vec3::length() const
{
    return sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
}

inline float vec3::squaredLength() const
{
    return e[0]*e[0]+e[1]*e[1]+e[2]*e[2];
}

inline void vec3::makeUnitVector()
{
    float k = 1.0 / (sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]));
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

inline vec3 unitVector(const vec3 v)
{
    return v / v.length();
}

// friend functions
inline std::istream& operator>>(std::istream &is, vec3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator <<(std::ostream &os, const vec3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;    
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]); 
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]); 
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]); 
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]); 
}

inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline vec3 operator*(const vec3 &v, float t)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline vec3 operator/(const vec3 &v, float t)
{
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}

inline float dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3((v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

