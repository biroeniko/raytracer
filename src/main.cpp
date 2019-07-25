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

#include <sys/types.h>
#include <sys/stat.h>

// STB IMAGE FOR WRITING IMAGE FILES
#ifndef STB_IMAGE_IMPLEMENTATION 
  #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
#endif /* STB_IMAGE_IMPLEMENTATION */

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION 
  #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
#endif /* STB_IMAGE_WRITE_IMPLEMENTATION */

// INCLUDE
#include "hitables/sphere.h"
#include "hitables/hitablelist.h"
#include "util/camera.h"
#include "materials/material.h"
#include "util/renderer.h"
#include "util/window.h"
#include "util/common.h"
#include "util/globals.h"
#include "util/scene.h"
#include "util/params.h"

#ifdef CUDA_ENABLED
    void initializeWorldCuda(lParams& lParams,
                             rParams& rParams);
    void destroyWorldCuda(lParams& lParams,
                          rParams& rParams);
#endif // CUDA_ENABLED


// Initialise the world.
void initializeWorld(lParams& lParams,
                     rParams& rParams)
{

#ifdef CUDA_ENABLED
    initializeWorldCuda(lParams, rParams);
#else
    rParams.image.reset(new Image(lParams.showWindow,
                                  lParams.writeImagePPM || lParams.writeImagePNG,
                                  nx, ny, tx, ty));
    rParams.cam.reset(new Camera(lookFrom, lookAt,
                                 vup, 20.0f, float(nx)/float(ny),
                                 distToFocus, aperture));
    rParams.renderer.reset(new Renderer(lParams.showWindow,
                                        lParams.writeImagePPM,
                                        lParams.writeImagePNG));
    rParams.world.reset(simpleScene2());

    if (lParams.showWindow)
    {
        rParams.w.reset(new Window(rParams.cam, rParams.renderer,
                                   nx, ny,
                                   thetaInit, phiInit,
                                   zoomScale,
                                   stepScale));
    }
#endif // CUDA_ENABLED
}

// Invoke the main renderer function.
void invokeRenderer(lParams& lParams,
                    rParams& rParams)
{
    std::ofstream ppmImageStream;

    if (lParams.writeImagePPM)
    {
        ppmImageStream.open("test.ppm");
        if (ppmImageStream.is_open())
            ppmImageStream << "P3\n" << nx << " " << ny << "\n255\n";
        else std::cout << "Unable to open file" << std::endl;
    }

    if (lParams.writeEveryImageToFile)
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
    }

    // If denoising is enabled, use the sample size for the denoising.
    #ifdef OIDN_ENABLED
       int numberOfIterations = (nsDenoise + nsBatch - 1)/nsBatch;
    #else
       int numberOfIterations = (ns + nsBatch - 1)/nsBatch;

    #endif // OIDN_ENABLED

    if (lParams.showWindow)
    {
        int j = 1;
        for (int i = 0; ; i++, j+=nsBatch)
        {
            rParams.w->updateImage(lParams, rParams, i+1);
            rParams.w->pollEvents(rParams.image);
            if (lParams.writeEveryImageToFile &&
                 #ifdef OIDN_ENABLED
                    (j >= sampleNrToWriteDenoise)
                 #else
                    (j >= sampleNrToWrite)
                 #endif // OIDN_ENABLED
                )
            {
                if (lParams.moveCamera)
                    rParams.w->moveCamera(rParams.image, rParams.image->fileOutputImage);
                j = 0;
            }
            if (rParams.w->refresh)
            {
                std::string currentFileName(folderName + "/" + fileName);
                currentFileName += formatNumber(imageNr);
                imageNr++;
                currentFileName += ".png";
                // Write png.
                stbi_write_png(currentFileName.c_str(), nx, ny, 3, rParams.image->fileOutputImage, nx * 3);

                i = -1;
                rParams.w->refresh = false;
            }
            if (rParams.w->quit)
                break;
        }
        std::cout << "Done." << std::endl;
    }
    else
    {
        for (int i = 0; i < numberOfIterations; i++)
            rParams.renderer->traceRays(rParams, i+1);
        std::cout << "Done." << std::endl;
    }

    // Write the files after the windows is closed.
    if (lParams.writeImagePPM)
    {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                ppmImageStream << int(rParams.image->fileOutputImage[(j*nx+i)*3]) << " "
                               << int(rParams.image->fileOutputImage[(j*nx+i)*3+1]) << " "
                               << int(rParams.image->fileOutputImage[(j*nx+i)*3+2]) << "\n";
        ppmImageStream.close();
    }

    if (lParams.writeImagePNG)
    {
        // Write png.
        stbi_write_png("test.png", nx, ny, 3, rParams.image->fileOutputImage, nx * 3);
    }
}

void raytrace(lParams lParams)
{
    rParams rParams;

    initializeWorld(lParams, rParams);
    invokeRenderer(lParams, rParams);
    #ifdef CUDA_ENABLED
        destroyWorldCuda(lParams, rParams);
    #endif // CUDA_ENABLED
}

int main(int argc, char **argv)
{

    bool runBenchmark = false;

    bool showWindow = true;
    bool writeImagePPM = true;
    bool writeImagePNG = true;
    bool writeEveryImageToFile = true;
    bool moveCamera = false;

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

            lParams lParams(false, false, false, false, false);
            raytrace(lParams);

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
        // Invoke renderer.
        lParams lParams(showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile, moveCamera);
        raytrace(lParams);
    }

    return 0;
}
