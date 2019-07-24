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

#ifdef CUDA_ENABLED
    void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG,
                             hitable*** list,
                             std::unique_ptr<hitable>& world,
                             std::unique_ptr<Window>& w,
                             std::unique_ptr<Image>& image,
                             std::unique_ptr<Camera>& cam,
                             std::unique_ptr<Renderer>& renderer);
    void destroyWorldCuda(bool showWindow, hitable** list,
                          std::unique_ptr<hitable>& world,
                          std::unique_ptr<Window>& w,
                          std::unique_ptr<Image>& image,
                          std::unique_ptr<Camera>& cam,
                          std::unique_ptr<Renderer>& renderer);
#else
    void initializeWorld(bool showWindow, bool writeImagePPM, bool writeImagePNG,
                         std::unique_ptr<hitable>& world,
                         std::unique_ptr<Window>& w,
                         std::unique_ptr<Image>& image,
                         std::unique_ptr<Camera>& cam,
                         std::unique_ptr<Renderer>& renderer)
    {
        image.reset(new Image(showWindow, writeImagePPM || writeImagePNG, nx, ny, tx, ty));
        cam.reset(new Camera(lookFrom, lookAt, vup, 20.0f, float(nx)/float(ny), distToFocus, aperture));
        renderer.reset(new Renderer(showWindow, writeImagePPM, writeImagePNG));
        world.reset(simpleScene2());

        if (showWindow)
            w.reset(new Window(cam, renderer, nx, ny, thetaInit, phiInit, zoomScale, stepScale));
    }
#endif // CUDA_ENABLED

void invokeRenderer(std::unique_ptr<hitable>& world,
                    std::unique_ptr<Window>& w,
                    std::unique_ptr<Image>& image,
                    std::unique_ptr<Camera>& cam,
                    std::unique_ptr<Renderer>& renderer,
                    bool showWindow,
                    bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile,
                    bool moveCamera)
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
        //if (error != 0)
        //    std::cerr << "Couldn't create output folder." << std::endl;
    }

    // If denoising is enabled, use the sample size for the denoising.
    #ifdef OIDN_ENABLED
       int numberOfIterations = (nsDenoise + nsBatch - 1)/nsBatch;
    #else
       int numberOfIterations = (ns + nsBatch - 1)/nsBatch;

    #endif // OIDN_ENABLED

    if (showWindow)
    {
        int j = 1;
        for (int i = 0; ; i++, j+=nsBatch)
        {
            w->updateImage(showWindow, writeImagePPM, writeImagePNG, ppmImageStream, w, cam, world, image, i+1, image->fileOutputImage);
            w->pollEvents(image, image->fileOutputImage);
            if (writeEveryImageToFile &&
                 #ifdef OIDN_ENABLED
                    (j >= sampleNrToWriteDenoise)
                 #else
                    (j >= sampleNrToWrite)
                 #endif // OIDN_ENABLED
                )
            {
                if (moveCamera)
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
            renderer->traceRays(cam, world, image, i+1);
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

struct renderingParams
{

};

void raytrace(bool showWindow, bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile, bool moveCamera)
{
    std::unique_ptr<Window> w;
    std::unique_ptr<Image> image;
    std::unique_ptr<Camera> cam;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<hitable> world;

    hitable** list;

    #ifdef CUDA_ENABLED
        initializeWorldCuda(showWindow, writeImagePPM, writeImagePNG, &list, world, w, image, cam, renderer);
        invokeRenderer(world, w, image, cam, renderer, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile, moveCamera);
        destroyWorldCuda(showWindow, list, world, w, image, cam, renderer);

        image.release();
        cam.release();
        renderer.release();
        world.release();

    #else
        initializeWorld(showWindow, writeImagePPM, writeImagePNG, world, w, image, cam, renderer);
        invokeRenderer(world, w, image, cam, renderer, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile, moveCamera);
    #endif // CUDA_ENABLED
}

int main(int argc, char **argv)
{
    bool writeImagePPM = true;
    bool writeImagePNG = true;
    bool showWindow = true;
    bool runBenchmark = false;
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
            raytrace(false, false, false, false, false);

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
        raytrace(showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile, moveCamera);
    }

    return 0;
}
