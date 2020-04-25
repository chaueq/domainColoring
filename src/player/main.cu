#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <unistd.h>
#include <sys/resource.h>
#include <chrono>
#include <X11/Xlib.h>
#include "render.cuh"

#define DEFAULT_GUI_WIDTH 1920
#define DEFAULT_GUI_HEIGHT 1080

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  if(argc < 2 || argc > 5){
    cerr << argv[0] << " <prv_file> [start_time] [<width> <height>] " << endl;
    return 1;
  }

  const char* inputPath = argv[1];
  if(strcmp(strrchr(inputPath, '.'), ".prv")){
    cerr << "File has other extension than .prv" << endl;
    return 1;
  }

  uint64_t time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
  const uintmax_t startTime = (argc == 3 || argc == 5) ? atol(argv[2])*1000 : time;
  const uint16_t guiWidth = (argc == 4 || argc == 5) ? atol((argc==4) ? argv[2] : argv[3]) : DEFAULT_GUI_WIDTH;
  const uint16_t guiHeight = (argc == 4 || argc == 5) ? atol((argc==4) ? argv[3] : argv[4]) : DEFAULT_GUI_HEIGHT;
  FILE* input = fopen(inputPath, "r");
  uint8_t headerBuf[13];
  if(fread(headerBuf, sizeof(uint8_t), 13, input) != 13 || ftell(input) != 13){
    cerr << "Failed reading file headers" << endl;
    return 1;
  }
  const uint64_t length =
    ((uint64_t)headerBuf[0] << 0) +
    ((uint64_t)headerBuf[1] << 8) +
    ((uint64_t)headerBuf[2] << 16) +
    ((uint64_t)headerBuf[3] << 24) +
    ((uint64_t)headerBuf[4] << 32) +
    ((uint64_t)headerBuf[5] << 40) +
    ((uint64_t)headerBuf[6] << 48) +
    ((uint64_t)headerBuf[7] << 56);
  const uint8_t fps = headerBuf[8];
  const uint16_t width = ((uint16_t)headerBuf[10] << 8) + (uint16_t)headerBuf[9];
  const uint16_t height = ((uint16_t)headerBuf[12] << 8) + (uint16_t)headerBuf[11];
  cuda::GpuMat frame(height, width, CV_8UC3);
  Mat gui(height, width, CV_8UC3), guiResized(guiHeight, guiWidth, CV_8UC3);
  uint8_t* values;
  uint8_t* repetitions;
  uint8_t* toBeLoaded;
  uint8_t key = 0;
  cudaMallocManaged(&values, width*height*3*sizeof(uint8_t));
  cudaMallocManaged(&repetitions, width*height*sizeof(uint8_t));
  cudaMallocManaged(&toBeLoaded, width*height*sizeof(uint8_t));
  uint32_t* readBuf = new uint32_t[width*height];
  uint64_t frameTime, waitTime;

  namedWindow(argv[0], WINDOW_NORMAL);
  setWindowProperty(argv[0], WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

  //load initial data
  fread(readBuf, sizeof(uint32_t), width*height, input);
  for(uint16_t x = 0; x < width; ++x)
  {
    for(uint16_t y = 0; y < height; ++y)
    {
      uint64_t i = ((y*width) + x);
      values[3*((x*height)+y) + 0] = (readBuf[i] >> 0) % 256;
      values[3*((x*height)+y) + 1] = (readBuf[i] >> 8) % 256;
      values[3*((x*height)+y) + 2] = (readBuf[i] >> 16) % 256;
      repetitions[(x*height)+y] = (readBuf[i] >> 24) % 256;
      toBeLoaded[(x*height)+y] = 0;
    }
  }
  for(uint64_t frameId = 0; frameId < length; ++frameId)
  {
    render(frame, values, repetitions, toBeLoaded);

    uint64_t amountToLoad = 0;
    for(uint16_t x = 0; x < width; ++x)
      for(uint16_t y = 0; y < height; ++y)
        amountToLoad += toBeLoaded[(x*height)+y];
    fread(readBuf, sizeof(uint32_t), amountToLoad, input);
    uintmax_t i = 0;
    for(uint16_t y = 0; y < height; ++y)
      for(uint16_t x = 0; x < width; ++x)
        if(toBeLoaded[(x*height)+y] == 1){
          values[3*((x*height)+y) + 0] = (readBuf[i] >> 0) % 256;
          values[3*((x*height)+y) + 1] = (readBuf[i] >> 8) % 256;
          values[3*((x*height)+y) + 2] = (readBuf[i] >> 16) % 256;
          repetitions[(x*height)+y] = (readBuf[i] >> 24) % 256;
          toBeLoaded[(x*height)+y] = 0;
          ++i;
        }

    time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    frameTime = startTime + ((1000*frameId) / fps);
    if(frameTime > time)
      waitTime = frameTime - time;
    else
      waitTime = 1;
    key = waitKey(waitTime);
    if(key == 27)
      break;

    frame.download(gui);
    cv::resize(gui, guiResized, Size(guiWidth, guiHeight));
    imshow(argv[0], guiResized);
  }
  time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

  delete readBuf;
  cudaFree(values);
  cudaFree(repetitions);
  cudaFree(toBeLoaded);

  destroyAllWindows();

  return 0;
}
