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

#include <iostream>
#include <fstream>
#include <float.h>
#include <random>
#include <chrono>
#include <SDL2/SDL.h>

#ifndef STB_IMAGE_IMPLEMENTATION 
  #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
#endif /* STB_IMAGE_IMPLEMENTATION */

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION 
  #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
#endif /* STB_IMAGE_WRITE_IMPLEMENTATION */

#include "hitables/sphere.h"
#include "hitables/hitableList.h"
#include "util/camera.h"
#include "materials/material.h"
#include "util/renderer.h"
#include "util/window.h"
#include "util/common.h"
#include "util/scene.h"

#include <sys/types.h>
#include <sys/stat.h>

#ifdef CUDA_ENABLED
    void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable*** list, hitable** world, Window** w, Image** image, Camera** cam, Renderer** render);
    void destroyWorldCuda(bool showWindow, hitable** list, hitable* world, Window* w, Image* image, Camera* cam, Renderer* render);
#else
    void initializeWorld(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable** world, Window** w, Image** image, Camera** cam, Renderer** render)
    {
        *image = new Image(showWindow, writeImagePPM || writeImagePNG, nx, ny, tx, ty);
        vec3 lookFrom(13.0f, 2.0f, 3.0f);
        vec3 lookAt(0.0f, 0.0f, 0.0f);
        *cam = new Camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), distToFocus);
        *render = new Renderer(showWindow, writeImagePPM, writeImagePNG);
        *world = simpleScene();

        if (showWindow)
            *w = new Window(*cam, *render, nx, ny, thetaInit, phiInit, zoomScale, stepScale);
    }
#endif // CUDA_ENABLED

void invokeRenderer(hitable* world, Window* w, Image* image, Camera* cam, Renderer* render, bool showWindow, bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile)
{
    std::ofstream ppmImageStream;

    if (writeImagePPM)
    {
        ppmImageStream.open("test.ppm");
        if (ppmImageStream.is_open())
            ppmImageStream << "P3\n" << nx << " " << ny << "\n255\n";
        else std::cout << "Unable to open file" << std::endl;
    }

    if (writeEveryImageToFile)
    {
        // Make the folder
        std::string path = "./" + folderName;
        mode_t mode = 0733; // UNIX style permissions
        int error = 0;
        #if defined(_WIN32)
            error = _mkdir(path.c_str()); // can be used on Windows
        #else
            error = mkdir(path.c_str(), mode);
        #endif
        if (error != 0)
            std::cerr << "Couldn't create output folder." << std::endl;
    }

    // If denoising is enabled, use the sample size for the denoising.
    #ifdef OIDN_ENABLED
       int numberOfIterations = nsDenoise;
    #else
        int numberOfIterations = ns;

    #endif // OIDN_ENABLED

    if (showWindow)
    {
        int j = 1;
        for (int i = 0; i < numberOfIterations; i++, j++)
        {
            w->updateImage(showWindow, writeImagePPM, writeImagePNG, ppmImageStream, w, cam, world, image, i+1, image->fileOutputImage);
            w->pollEvents(image, image->fileOutputImage);
            if (writeEveryImageToFile && (j == sampleNrToWrite))
            {
                w->moveCamera(image, image->fileOutputImage);
                j = 0;
            }
            if (w->refresh)
            {
                std::string currentFileName(folderName + "/" + fileName);
                currentFileName += formatNumber(imageNr);
                imageNr++;
                currentFileName += ".png";
                // write png
                stbi_write_png(currentFileName.c_str(), nx, ny, 3, image->fileOutputImage, nx * 3);

                i = -1;
                w->refresh = false;
            }
            if (w->quit)
                break;
        }
        std::cout << "Done." << std::endl;
    }
    else
    {
        for (int i = 0; i < numberOfIterations; i++)
            render->traceRays(nullptr, cam, world, image, i+1, image->fileOutputImage);
        std::cout << "Done." << std::endl;
    }

    // we write the files after the windows is closed
    if (writeImagePPM)
    {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                ppmImageStream << int(image->fileOutputImage[(j*nx+i)*3]) << " " << int(image->fileOutputImage[(j*nx+i)*3+1]) << " " << int(image->fileOutputImage[(j*nx+i)*3+2]) << "\n";
        ppmImageStream.close();
    }

    if (writeImagePNG)
    {
        // write png
        stbi_write_png("test.png", nx, ny, 3, image->fileOutputImage, nx * 3);
    }
}

void raytrace(bool showWindow, bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile)
{
    Window* w;
    Image* image;
    Camera* cam;
    Renderer* render;
    hitable* world;

    hitable** list;

    #ifdef CUDA_ENABLED
        initializeWorldCuda(showWindow, writeImagePPM, writeImagePNG, &list, &world, &w, &image, &cam, &render);
        invokeRenderer(world, w, image, cam, render, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);
        destroyWorldCuda(showWindow, list, world, w, image, cam, render);

    #else
        initializeWorld(showWindow, writeImagePPM, writeImagePNG, &world, &w, &image, &cam, &render);
        invokeRenderer(world, w, image, cam, render, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);

        delete image;
        delete cam;
        delete render;
        delete world;
        if (showWindow)
            delete w;
    #endif // CUDA_ENABLED
}

int main(int argc, char **argv)
{
    bool writeImagePPM = true;
    bool writeImagePNG = true;
    bool showWindow = false;
    bool runBenchmark = false;
    bool writeEveryImageToFile = false;

    // Run benchmark.
    if (runBenchmark)
    {
        std::ofstream benchmarkStream;

        for (int i = 0; i < benchmarkCount; i++)
        {
            benchmarkStream.open("../benchmark/benchmarkResultCUDA.txt", std::ios_base::app);
            // Record start time
            auto start = std::chrono::high_resolution_clock::now();

            // Invoke renderer
            raytrace(false, false, false, false);

            // Record end time
            auto finish = std::chrono::high_resolution_clock::now();

            // Compute elapsed time
            std::chrono::duration<double> elapsed = finish - start;

            // Write results to file
            benchmarkStream << ns << " " <<  elapsed.count() << "s\n";

            benchmarkStream.close();

        }
    }
    // Run code without benchmarking.
    else
    {
        // Invoke renderer
        raytrace(showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);
    }

    return 0;
}
