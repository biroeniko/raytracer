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

#include <memory>
#include "util/window.h"

// Rendering parameters.
class rParams
{

    public:

        std::unique_ptr<Window> w;
        std::unique_ptr<Image> image;
        std::unique_ptr<Camera> cam;
        std::unique_ptr<Renderer> renderer;
        std::unique_ptr<hitable> world;

        hitable** list;

        ~rParams()
        {
            image.release();
            cam.release();
            renderer.release();
            world.release();
        }

};

// Logistical parameters.
class lParams
{

    public:

        bool showWindow;
        bool writeImagePPM;
        bool writeImagePNG;
        bool writeEveryImageToFile;
        bool moveCamera;

        lParams(bool showWindow,
                bool writeImagePPM,
                bool writeImagePNG,
                bool writeEveryImageToFile,
                bool moveCamera) :
                showWindow(showWindow),
                writeImagePPM(writeImagePPM),
                writeImagePNG(writeImagePNG),
                writeEveryImageToFile(writeEveryImageToFile),
                moveCamera(moveCamera)
        {

        }

};

