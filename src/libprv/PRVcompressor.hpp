#ifndef PRV_COMPRESSOR_HPP
#define PRV_COMPRESSOR_HPP

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <stdio.h>
#include <sys/resource.h>
#include "RGB.hpp"
#include "Pixel.hpp"
#include "reading.hpp"
#include "writing.hpp"
#include "checking.hpp"

using namespace std;
using namespace cv;

class PRVcompressor
{
  public:
    PRVcompressor(const char* outputPath, uint8_t tollerance, uint8_t fps, uint64_t length, uint16_t width, uint16_t height);
    void upload(Mat frame);
    ~PRVcompressor();
  private:
    uint8_t THREAD_NUM;
    uint8_t tollerance;

    FILE* output;
    uint8_t fps;
    uint64_t length;
    uint16_t width;
    uint16_t height;

    Pixel* data;
    uint8_t* repetitions;
    void* status;
    uint8_t readingDone;
    pthread_t cui, reading, writing;
    readingWorkerData readingData;
    writingWorkerData writingData;

    uint64_t readingFrameId;
    uint64_t writingFrameId;
    pthread_t* threads;

    readingWorkerData* readingTD;
    writingWorkerData* writingTD;
    checkingWorkerData* checkingTD;
};

#endif
