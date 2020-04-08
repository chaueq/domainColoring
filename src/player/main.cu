#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <unistd.h>
#include <sys/resource.h>
#include <chrono>
#include "render.cuh"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  if(argc != 5){
    cerr << argv[0] << " <file> <start_time> <width> <height>" << endl;
    return 1;
  }

  const char* inputPath = argv[1];
  const uintmax_t startTime = atol(argv[2]);
  const uint16_t guiWidth = atol(argv[3]);
  const uint16_t guiHeight = atol(argv[4]);
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
  uint8_t*** values;
  uint8_t** repetitions;
  uint8_t** toBeLoaded;
  cudaMallocManaged(&values, width*sizeof(uint8_t**));
  cudaMallocManaged(&repetitions, width*sizeof(uint8_t*));
  cudaMallocManaged(&toBeLoaded, width*sizeof(uint8_t*));
  for(uint16_t x = 0; x < width; ++x)
  {
    cudaMallocManaged(&values[x], height*sizeof(uint8_t*));
    cudaMallocManaged(&repetitions[x], height*sizeof(uint8_t));
    cudaMallocManaged(&toBeLoaded[x], height*sizeof(uint8_t));
    for(uint16_t y = 0; y < height; ++y)
      cudaMallocManaged(&values[x][y], 3*sizeof(uint8_t));
  }
  uint32_t* readBuf = new uint32_t[width*height];

  namedWindow(argv[0], WINDOW_NORMAL);
  setWindowProperty(argv[0], WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

  //load initial data
  fread(readBuf, sizeof(uint32_t), width*height, input);
  for(uint16_t x = 0; x < width; ++x)
  {
    for(uint16_t y = 0; y < height; ++y)
    {
      uint64_t i = ((y*width) + x);
      values[x][y][0] = (readBuf[i] >> 0) % 256;
      values[x][y][1] = (readBuf[i] >> 8) % 256;
      values[x][y][2] = (readBuf[i] >> 16) % 256;
      repetitions[x][y] = (readBuf[i] >> 24) % 256;
      toBeLoaded[x][y] = 0;
    }
  }

  for(uint64_t frameId = 0; frameId < length; ++frameId)
  {
    render(frame, values, repetitions, toBeLoaded);

    uint64_t amountToLoad = 0;
    for(uint16_t x = 0; x < width; ++x)
      for(uint16_t y = 0; y < height; ++y)
        amountToLoad += toBeLoaded[x][y];
    fread(readBuf, sizeof(uint32_t), amountToLoad, input);
    uintmax_t i = 0;
    for(uint16_t y = 0; y < height; ++y)
      for(uint16_t x = 0; x < width; ++x)
        if(toBeLoaded[x][y] == 1){
          values[x][y][0] = (readBuf[i] >> 0) % 256;
          values[x][y][1] = (readBuf[i] >> 8) % 256;
          values[x][y][2] = (readBuf[i] >> 16) % 256;
          repetitions[x][y] = (readBuf[i] >> 24) % 256;
          toBeLoaded[x][y] = 0;
          ++i;
        }

    if(waitKey(1) == 27) //TODO: bigger precision
      break;
    frame.download(gui);
    // cv::resize(gui, guiResized, Size(guiWidth, guiHeight));
    imshow(argv[0], gui);
  }


  delete readBuf;
  cudaFree(values);
  cudaFree(repetitions);
  cudaFree(toBeLoaded);

  destroyAllWindows();

  return 0;
}
