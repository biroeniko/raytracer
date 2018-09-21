#pragma once

#include <iostream>
#include <fstream>
#include "vec3.h"

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