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

#define nx 1200
#define ny 600
#define ns 100          // sample size


struct window
{
    // x,y,w,h
    SDL_Window* SDLwindow;
    SDL_Renderer* renderer;
    SDL_Event event;
    const Uint8* keys;

    window()
    {
        SDLwindow = NULL; 
        SDL_Surface* screenSurface = NULL;
        if (SDL_Init(SDL_INIT_VIDEO) < 0) 
            std::cout << "SDL could not initialize! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;
        else 
        { 
            SDLwindow = SDL_CreateWindow("Ray tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, nx, ny, SDL_WINDOW_SHOWN); 
            if (SDLwindow == NULL) 
            { 
                std::cout << "Window could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
            renderer = SDL_CreateRenderer(SDLwindow, -1, SDL_RENDERER_SOFTWARE);
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
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderPresent(renderer);
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

    ~window()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(SDLwindow); 
        SDL_Quit();
    }
};


bool traceRays(bool showWindow, bool writeImage, window* w, hitable* world, uint8_t *image, camera& cam, std::ofstream& myfile)
{
    // j track rows - from top to bottom
    for (int j = ny-1; j >= 0; j--)
    {
        // i tracks columns - left to right
        for (int i = 0; i < nx; i++)
        {
            vec3 col(0.0f, 0.0f, 0.0f);
            #pragma omp parallel for
            for (int s = 0; s < ns; s++)
            {
                float u = float(i + dist(mt)) / float(nx); // left to right
                float v = float(j + dist(mt)) / float(ny); // bottom to top
                
                ray r = cam.getRay(u,v);

                col += color(r, world, 0);
            }
            col /= float(ns);
            
            // Gamma encoding of images is used to optimize the usage of bits 
            // when encoding an image, or bandwidth used to transport an image, 
            // by taking advantage of the non-linear manner in which humans perceive 
            // light and color. (wikipedia)
            
            // we use gamma 2: raising the color to the power 1/gamma (1/2)
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int ir = int(255.99f*col[0]);
            int ig = int(255.99f*col[1]);
            int ib = int(255.99f*col[2]);

            if (writeImage)
            {
                // PNG
                int index = (ny - 1 - j) * nx + i;
                int index3 = 3 * index;

                image[index3 + 0] = ir;
                image[index3 + 1] = ig;
                image[index3 + 2] = ib;
                myfile << ir << " " << ig << " " << ib << "\n";
            }

            if (showWindow)
            {
                SDL_SetRenderDrawColor(w->renderer, ir, ig, ib, 255);
                SDL_RenderDrawPoint(w->renderer, i, ny - j - 1);
            }
        }

        if (showWindow)
        {
            SDL_RenderPresent(w->renderer);
            if (w->quit())
                return false;
        }
    }

    return true;
}

void draw(bool showWindow, bool writeImage)
{
    window* w;

    if (showWindow)
    {
        w = new window;
    }

    uint8_t *image;
    std::ofstream myfile;
    if (writeImage)
    {
        // for png file
        image = new uint8_t[nx * ny * 3];
        myfile.open("test.ppm");
        if (myfile.is_open())
            myfile << "P3\n" << nx << " " << ny << "\n255\n";
        else std::cout << "Unable to open file" << std::endl;
    }

    hitable *world = randomScene();

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
        traceRays(showWindow, writeImage, w, world, image, cam, myfile);    
        
        while (!w->quit())
        {
            // wait for user input
        }
    }
    else
        traceRays(showWindow, writeImage, w, world, image, cam, myfile);

    if (showWindow)
    {
        delete w;
    }
    
    if (writeImage)
    {
        myfile.close();

        // write png
        stbi_write_png("test.png", nx, ny, 3, image, nx * 3);
        delete[] image;
    }
}

int main()
{
	bool writeImage = true;
    bool showWindow = true;

    draw(showWindow, writeImage);
    return 0;
}