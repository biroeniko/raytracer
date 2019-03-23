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

#include "util/vec3.h"

const int nx = 1280;
const int ny = 720;
const int ns = 64;                     // sample size
const int nsDenoise = 64;
static int imageNr = 0;
const int sampleNrToWrite = 16;
const int sampleNrToWriteDenoise = sampleNrToWrite;
const std::string folderName = "output";
const std::string fileName = "raytracer";
#ifdef CUDA_ENABLED
    const int nsBatch = 4;
#else
    const int nsBatch = 1;
#endif
const int tx = 16;                      // block size
const int ty = 16;
const int benchmarkCount = 100;
const float thetaInit = 1.34888f;
const float phiInit = 1.32596f;
const float zoomScale = 0.5f;
const float stepScale = 0.5f;

extern vec3 lookFrom;
extern vec3 lookAt;

const float distToFocus = 10.0f;
const float aperture = 0.1f;
