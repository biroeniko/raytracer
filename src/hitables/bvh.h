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

class bvhNode : public hitable
{

    public:

        bvhNode() {}
        bvhNode(hitable **l, int n, float t0, float t1);
        virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
        virtual bool boundingBox(float t0, float t1, aabb& box) const override;
        hitable *left;
        hitable *right;
        aabb box;

};

bool bvhNode::boundingBox(float t0, float t1, aabb& b) const
{
    b = box;
    return true;
}


bool bvhNode::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const
{

    if (box.hit(r, tMin, tMax))
    {
        hitRecord leftRec, rightRec;
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


