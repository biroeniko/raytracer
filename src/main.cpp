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
#define thetaInit 1.34888
#define phiInit 1.32596

/*
class SystemManager 
{
    public:
        bool running;
        SDL_Event events;
        const Uint8* keys;
        void inputManager(); 

        SystemManager()
        {
            keys = SDL_GetKeyboardState(NULL);
            if (keys == NULL)
            { 
                std::cout << "Keys could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
        }
};

void SystemManager::inputManager() 
{
    while(SDL_PollEvent(&events)) 
    {
        if(events.type == SDL_QUIT || keys[SDL_SCANCODE_ESCAPE])
            running = false;
    }
}
*/


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

struct Window;

bool traceRays(bool showWindow, bool writeImagePPM, bool writeImagePNG, std::ofstream& myfile, Window* w, camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage);

struct Window
{
    // x,y,w,h
    SDL_Rect SDLWindowRect = { 0, 0, nx, ny };
    SDL_Window* SDLWindow;
    SDL_Renderer* SDLRenderer;
    SDL_Texture* SDLTexture;

    Uint32 *windowPixels;   

    bool quit;
    bool mouseDragIsInProgress;
    bool refresh;

	float theta;
	float phi;
	const float delta = 0.1 * M_PI / 180.0f;

    camera* windowCamera;

    Window(camera* cam): windowCamera(cam)
    {

	    theta = thetaInit;
	    phi = phiInit;

        quit = false;
        mouseDragIsInProgress = false;
        refresh = false;

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
            SDLRenderer = SDL_CreateRenderer(SDLWindow, -1, SDL_RENDERER_SOFTWARE);
            if (SDLRenderer == NULL) 
            { 
                std::cout << "Renderer could not be created! SDL_Error: %s\n" <<  SDL_GetError() << std::endl;; 
            }
        }

        SDL_RenderSetLogicalSize(SDLRenderer, SDLWindowRect.w, SDLWindowRect.h);
        SDL_SetRenderDrawColor(SDLRenderer, 0, 0, 0, 255);
        SDL_RenderClear(SDLRenderer);
        SDL_RenderPresent(SDLRenderer);

        SDLTexture = SDL_CreateTexture(SDLRenderer,
                                    SDL_PIXELFORMAT_ARGB8888,
                                    SDL_TEXTUREACCESS_STATIC,
                                    nx, ny);

        windowPixels = new Uint32[nx*ny];

        windowCamera->setLookFrom(theta, phi);
    }

    ~Window()
    {
        SDL_DestroyTexture(SDLTexture);
        SDL_DestroyRenderer(SDLRenderer);
        SDL_DestroyWindow(SDLWindow); 
        delete[] windowPixels;
        SDL_Quit();
    }

    void updateImage(bool showWindow, bool writeImagePPM, bool writeImagePNG, std::ofstream& myfile, Window* w, camera* cam, 
                        hitable* world, Image* image,  int sampleCount, uint8_t *fileOutputImage) 
    {
		    traceRays(showWindow, writeImagePPM, writeImagePNG, myfile, w, cam, world, image, sampleCount, fileOutputImage);    
            std::cout << "Sample nr. " << sampleCount << std::endl;
            SDL_UpdateTexture(w->SDLTexture, NULL, w->windowPixels, nx * sizeof(Uint32));
            SDL_RenderCopy(w->SDLRenderer, w->SDLTexture, NULL, NULL);
            SDL_RenderPresent(w->SDLRenderer);
	}

    bool pollEvents(Image* image, uint8_t *fileOutputImage)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch(event.type)
            {
                case SDL_MOUSEMOTION:
                    if (mouseDragIsInProgress)
                    {
                        int mx = event.motion.xrel;
                        int my = event.motion.yrel;
					    theta += -my * delta;
					    if (theta < delta) 
                            theta = delta;
					    if (theta > (M_PI_2 - delta)) 
                            theta = M_PI_2 - delta;
					    phi += -mx * delta;
					    windowCamera->setLookFrom(theta, phi);
                        for (int i = 0; i < nx*ny; i++)
	                    {
                            image->pixels[i/ny][i%ny] = vec3(0, 0, 0);
                            fileOutputImage = 0;
	                    }   
                        refresh = true;
                    }
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    {
				        mouseDragIsInProgress = true;
                    }
                    break;
			    case SDL_MOUSEBUTTONUP:
                    {
				        mouseDragIsInProgress = false;
                    }
				    break;
			    case SDL_QUIT:
				    quit = true;
				    break;
            }
        }
    }

    void waitQuit()
    {
        SDL_Event event;
        while (!quit)
        {
            SDL_WaitEvent(&event);
            quit = (event.type == SDL_QUIT);
        }
    }
};


bool traceRays(bool showWindow, bool writeImagePPM, bool writeImagePNG, std::ofstream& myfile, Window* w, camera* cam, hitable* world, Image* image, int sampleCount, uint8_t *fileOutputImage)
{
    // collapses the two nested fors into the same parallel for
    #pragma omp parallel for collapse(2)
    // j track rows - from top to bottom
    for (int j = 0; j < ny; j++)
    {
        // i tracks columns - left to right
        for (int i = 0; i < nx; i++)
        {
            float u = float(i + dist(mt)) / float(nx); // left to right
            float v = float(j + dist(mt)) / float(ny); // bottom to top
                
            ray r = cam->getRay(u,v);

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

void invokeRenderer(bool showWindow, bool writeImagePPM, bool writeImagePNG)
{
    Window* w;
    Image* image = new Image(nx, ny);

    //hitable *world = randomScene();
    hitable *world = simpleScene2();

    vec3 lookFrom(13.0f, 2.0f, 3.0f);
    vec3 lookAt(0.0f, 0.0f, 0.0f);
    float distToFocus = 10.0f;
    float aperture = 0.1f;

    camera* cam = new camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), aperture, distToFocus);

    if (showWindow)
    {
        w = new Window(cam);
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
    
    // create source of randomness, and initialize it with non-deterministic seed
    std::random_device r;
    std::mt19937 mt(r());
    // a distribution that takes randomness and produces values in specified range
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (showWindow)
    {
        for (int i = 0; i < ns; i++)
        {
            w->updateImage(showWindow, writeImagePPM, writeImagePNG, myfile, w, cam, world, image, i+1, fileOutputImage);
			w->pollEvents(image, fileOutputImage);
            if (w->refresh)
            {
                i = -1;
                w->refresh = false;
            }
            if (w->quit)
                break;
        }
        std::cout << "Done." << std::endl;

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

    invokeRenderer(showWindow, writeImagePPM, writeImagePNG);
    return 0;
}