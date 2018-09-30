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
#include "util/scene.h"

#define nx 1400
#define ny 700
#define ns 100          // sample size

struct Window
{
    // x,y,w,h
    SDL_Rect windowRect = { 0, 0, nx, ny };
    SDL_Window* SDLWindow;
    SDL_Renderer* renderer;
    SDL_Texture* texture;

    Uint32 *windowPixels;   
    SDL_Event event;
    const Uint8* keys;

    Window()
    {
        SDLWindow = NULL; 
        SDL_Surface* screenSurface = NULL;
        if (SDL_Init(SDL_INIT_VIDEO) < 0) 
            std::cout << "SDL could not initialize! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;
        else 
        { 
            SDLWindow = SDL_CreateWindow("Ray tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, nx, ny, SDL_WINDOW_SHOWN); 
            if (SDLWindow == NULL) 
            { 
                std::cout << "Window could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
            renderer = SDL_CreateRenderer(SDLWindow, -1, SDL_RENDERER_SOFTWARE);
            if (renderer == NULL) 
            { 
                std::cout << "Renderer could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
            keys = SDL_GetKeyboardState(NULL);
            if (keys == NULL)
            { 
                std::cout << "Keys could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
        }

        SDL_RenderSetLogicalSize(renderer, windowRect.w, windowRect.h);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderPresent(renderer);

        texture = SDL_CreateTexture(renderer,
                                    SDL_PIXELFORMAT_ARGB8888,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    nx, ny);

        windowPixels = new Uint32[nx*ny];
    }

    bool quit() 
    {
        // Get the current pressed key
        SDL_PollEvent(&event);

        if (keys[SDL_SCANCODE_ESCAPE] || event.type == SDL_QUIT)
            return true;
        else
            return false;
    }

    ~Window()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(SDLWindow); 
        SDL_Quit();
    }
};

struct Image
{
    vec3** pixels;
    int rows;
    int columns;

    Image(int x, int y) : rows(x), columns(y)
    {
        pixels = new vec3*[rows];
        for (int i = 0; i < rows; i++)
            pixels[i] = new vec3[columns];
    }

    ~Image()
    {
        for (int i = 0; i < rows; ++i)
            delete [] pixels[i];
        delete [] pixels;
    }
};


bool traceRays(bool showWindow, bool writeImagePPM, bool writeImagePNG, std::ofstream& myfile, Window* w, camera& cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage)
{
    // collapses the two nested fors into the same parallel for
    #pragma omp parallel for collapse(2)
    // j track rows - from top to bottom
    for (int j = ny-1; j >= 0; j--)
    {
        // i tracks columns - left to right
        for (int i = 0; i < nx; i++)
        {
            float u = float(i + dist(mt)) / float(nx); // left to right
            float v = float(j + dist(mt)) / float(ny); // bottom to top
                
            ray r = cam.getRay(u,v);

            image->pixels[i][j] += color(r, world, 0);

            vec3 col = image->pixels[i][j] / sampleCount;
            
            // Gamma encoding of images is used to optimize the usage of bits 
            // when encoding an image, or bandwidth used to transport an image, 
            // by taking advantage of the non-linear manner in which humans perceive 
            // light and color. (wikipedia)
            
            // we use gamma 2: raising the color to the power 1/gamma (1/2)
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int ir = int(255.99f*col[0]);
            int ig = int(255.99f*col[1]);
            int ib = int(255.99f*col[2]);
            
            if (writeImagePNG)
            {
                // PNG
                int index = (ny - 1 - j) * nx + i;
                int index3 = 3 * index;

                fileOutputImage[index3 + 0] = ir;
                fileOutputImage[index3 + 1] = ig;
                fileOutputImage[index3 + 2] = ib;
            }

            if (showWindow)
                w->windowPixels[(ny-j-1)*nx + i] = (ir << 16) | (ig << 8) | (ib);
        }
    }

    return true;
}

void draw(bool showWindow, bool writeImagePPM, bool writeImagePNG)
{
    Window* w;
    Image* image = new Image(nx, ny);

    if (showWindow)
    {
        w = new Window;
    }

    uint8_t *fileOutputImage;
    std::ofstream myfile;
    
    if (writeImagePNG || writeImagePPM)
    {
        // for png file
        fileOutputImage = new uint8_t[nx * ny * 3];
    }
    
    if (writeImagePPM)
    {
        myfile.open("test.ppm");
        if (myfile.is_open())
            myfile << "P3\n" << nx << " " << ny << "\n255\n";
        else std::cout << "Unable to open file" << std::endl;
    }

    //hitable *world = randomScene();
    hitable *world = simpleScene2();

    vec3 lookFrom(13.0f, 2.0f, 3.0f);
    vec3 lookAt(0.0f, 0.0f, 0.0f);
    float distToFocus = 10.0f;
    float aperture = 0.1;

    camera cam(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), aperture, distToFocus);
    
    // create source of randomness, and initialize it with non-deterministic seed
    std::random_device r;
    std::mt19937 mt(r());
    // a distribution that takes randomness and produces values in specified range
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (showWindow)
    {
        for (int i = 0; i < ns; i++)
        {
            traceRays(showWindow, writeImagePPM, writeImagePNG, myfile, w, cam, world, image, i+1, fileOutputImage);    
            std::cout << "Sample nr. " << i+1 << std::endl;
            if (w->quit())
                break;
            SDL_UpdateTexture(w->texture, NULL, w->windowPixels, nx * sizeof(Uint32));
            SDL_RenderCopy(w->renderer, w->texture, NULL, NULL);
            SDL_RenderPresent(w->renderer);
        }
        std::cout << "Done." << std::endl;

        while (!w->quit());

        // we write the files after the windows is closed
        if (writeImagePPM)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int i = 0; i < nx; i++)
                {
                    myfile << int(fileOutputImage[(j*nx+i)*3]) << " " << int(fileOutputImage[(j*nx+i)*3+1]) << " " << int(fileOutputImage[(j*nx+i)*3+2]) << "\n";
                }
            }
            myfile.close();
        }

        if (writeImagePNG)
        {
            // write png
            stbi_write_png("test.png", nx, ny, 3, fileOutputImage, nx * 3);
        }

        if (writeImagePNG || writeImagePPM)
            delete[] fileOutputImage;
    }
    else
    {
       for (int i = 0; i < ns; i++)
        {
            traceRays(showWindow, writeImagePPM, writeImagePNG, myfile, w, cam, world, image, i+1, fileOutputImage);    
            std::cout << "Sample nr. " << i+1 << std::endl;
        }
        std::cout << "Done." << std::endl;
    }
    if (showWindow)
    {
        delete w;
    }
}

int main()
{
	bool writeImagePPM = true;
    bool writeImagePNG = true;
    bool showWindow = true;

    draw(showWindow, writeImagePPM, writeImagePNG);
    return 0;
}