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

#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"

vec3 colorLerp(const ray& r)
{
    vec3 unitDirection = unitVector(r.direction());
    // -1.0 < y < 1.0 => 
    // 0.0 <  y < 2.0 =>
    // 0.0 <  y < 1.0
    float t = 0.5*(unitDirection.y() + 1.0);
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

void helloRays()
{
    int nx = 800;
    int ny = 600;

    std::ofstream myfile ("test.ppm");
    if (myfile.is_open())
    {
        myfile << "P3\n" << nx << " " << ny << "\n255\n";
        
        vec3 lowerLeftCorner(-2.0, -1.0, -1.0);
        vec3 horizontal(4.0, 0.0, 0.0);             // difference
        vec3 vertical(0.0, 2.0, 0.0);               // difference
        vec3 origin(0.0, 0.0, 0.0);

        // j track rows - from top to bottom
        for (int j = ny-1; j >= 0; j--)
        {
            // i tracks columns - left to right
            for (int i = 0; i < nx; i++)
            {
                float u = float(i) / float(nx); // left to right
                float v = float(j) / float(ny); // bottom to top
                
                ray r(origin, lowerLeftCorner + u*horizontal + v*vertical);

                vec3 col = colorLerp(r);
                
                int ir = int(255.99*col[0]);
                int ig = int(255.99*col[1]);
                int ib = int(255.99*col[2]);
                myfile << ir << " " << ig << " " << ib << "\n";
            }
        }
        myfile.close();
    }
    else std::cout << "Unable to open file";
}

void helloWorld()
{
    int nx = 800;
    int ny = 600;

    std::ofstream myfile ("test.ppm");
    if (myfile.is_open())
    {
        myfile << "P3\n" << nx << " " << ny << "\n255\n";
        
        // j track rows - from top to bottom
        for (int j = ny-1; j >= 0; j--)
        {
            // i tracks columns - left to right
            for (int i = 0; i < nx; i++)
            {
                vec3 col(float(float(i) / float(nx)), float(float(j) / float(ny)), 0.2); 
                // red goes from black to fully -> from left to right
                //float r = float(i) / float(nx);
                // green goes from black to fully -> from bottom to top
                //float g = float(j) / float(ny);
                // blue - experimental (original: 0.2) -> from left to right
                //float b = float(i) / float(ny);
                int ir = int(255.99*col[0]);
                int ig = int(255.99*col[1]);
                int ib = int(255.99*col[2]);
                myfile << ir << " " << ig << " " << ib << "\n";
            }
        }
        myfile.close();
    }
    else std::cout << "Unable to open file";
}