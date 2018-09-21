#include <iostream>
#include <fstream>

void testImage()
{
    int nx = 800;
    int ny = 600;

    std::ofstream myfile ("test.ppm");
    if (myfile.is_open())
    {
        myfile << "P3\n" << nx << " " << ny << "\n255\n";
        
        for (int j = ny-1; j >= 0; j--)
        {
            for (int i = 0; i < nx; i++)
            {
                float r = float(i) / float(nx);
                float g = float(j) / float(ny);
                float b = 0.2;
                int ir = int(255.99*r);
                int ig = int(255.99*g);
                int ib = int(255.99*b);
                myfile << ir << " " << ig << " " << ib << "\n";
            }
        }
        myfile.close();
    }
    else std::cout << "Unable to open file";
}