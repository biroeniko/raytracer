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

#include "hitable.h"
#include "util/randomgenerator.h"

class BVHNode : public Hitable
{

    public:

        CUDA_DEV BVHNode() {}
        CUDA_DEV BVHNode(Hitable **l, int n, float t0, float t1);
        CUDA_DEV virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
        CUDA_DEV virtual bool boundingBox(float t0, float t1, AABB& box) const override;

        Hitable *left;
        Hitable *right;
        AABB box;
        RandomGenerator rng;

};

inline CUDA_DEV bool BVHNode::boundingBox(float t0, float t1, AABB& b) const
{
    b = box;
    return true;
}

inline CUDA_DEV bool BVHNode::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{

    if (box.hit(r, tMin, tMax))
    {
        HitRecord leftRec, rightRec;
        bool hitLeft = left->hit(r, tMin, tMax, leftRec);
        bool hitRight = right->hit(r, tMin, tMax, rightRec);
        if (hitLeft && hitRight)
        {
            if (leftRec.time < rightRec.time)
                rec = leftRec;
            else
                rec = rightRec;
            return true;
        }
        else if (hitLeft)
        {
            rec = leftRec;
            return true;
        }
        else if (hitRight)
        {
            rec = rightRec;
            return true;
        }
        else
            return false;
    }

    return false;

}

inline CUDA_DEV int boxCompareX(const void* a, const void* b)
{

    AABB boxLeft, boxRight;
    Hitable* ah = *(Hitable**)a;
    Hitable* bh = *(Hitable**)b;

    #ifndef CUDA_ENABLED
        if (!ah->boundingBox(0,0, boxLeft) ||
            !bh->boundingBox(0,0, boxRight))
        {
            std::cerr << "No bounding box in bvhNode constructor" << std::endl;
        }
    #endif
    if (boxLeft.min().x() - boxRight.min().x() < 0.0)
        return -1;
    else
        return 1;

}

inline CUDA_DEV int boxCompareY(const void* a, const void* b)
{

    AABB boxLeft, boxRight;
    Hitable* ah = *(Hitable**)a;
    Hitable* bh = *(Hitable**)b;

    #ifndef CUDA_ENABLED
        if (!ah->boundingBox(0,0, boxLeft) ||
            !bh->boundingBox(0,0, boxRight))
        {
            std::cerr << "No bounding box in bvhNode constructor" << std::endl;
        }
    #endif

    if (boxLeft.min().y() - boxRight.min().y() < 0.0)
        return -1;
    else
        return 1;

}

inline CUDA_DEV int boxCompareZ(const void* a, const void* b)
{

    AABB boxLeft, boxRight;
    Hitable* ah = *(Hitable**)a;
    Hitable* bh = *(Hitable**)b;

    #ifndef CUDA_ENABLED
        if (!ah->boundingBox(0,0, boxLeft) ||
            !bh->boundingBox(0,0, boxRight))
        {
            std::cerr << "No bounding box in bvhNode constructor" << std::endl;
        }
    #endif
    if (boxLeft.min().z() - boxRight.min().z() < 0.0)
        return -1;
    else
        return 1;

}

inline CUDA_DEV BVHNode::BVHNode(Hitable **l, int n, float t0, float t1)
{

    int axis = int(3*rng.get1f());

    if (axis == 0)
       qsort(l, n, sizeof(Hitable *), boxCompareX);
    else if (axis == 1)
       qsort(l, n, sizeof(Hitable *), boxCompareY);
    else
       qsort(l, n, sizeof(Hitable *), boxCompareZ);
    if (n == 1)
        left = right = l[0];
    else if (n == 2)
    {
        left = l[0];
        right = l[1];
    }
    else
    {
        left = new BVHNode(l, n/2, t0, t1);
        right = new BVHNode(l + n/2, n - n/2, t0, t1);
    }

    AABB boxLeft, boxRight;
    #ifndef CUDA_ENABLED
        if (!left->boundingBox(t0,t1, boxLeft) || !right->boundingBox(t0, t1, boxRight))
            std::cerr << "No bounding box in bvhNode constructor" << std::endl;
    #endif
    box = surroundingBox(boxLeft, boxRight);

}
