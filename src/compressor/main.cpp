#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <stdio.h>
#include <sys/resource.h>
#include "RGB.hpp"
#include "Pixel.hpp"
#include "cui.hpp"
#include "reading.hpp"
#include "writing.hpp"
#include "checking.hpp"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
  if(argc != 4)
  {
    cerr << argv[0] << " <input_file> <output_file> <tollerance>" << endl;
    return 1;
  }

  const uint8_t THREAD_NUM = sysconf(_SC_NPROCESSORS_ONLN);
  const char* inputPath = argv[1];
  const char* outputPath = argv[2];
  const uint8_t tollerance = atoi(argv[3]);

  VideoCapture input(inputPath);
  FILE* output = fopen(outputPath, "w");
  const uint8_t fps = input.get(CAP_PROP_FPS);
  const uint64_t length = input.get(CAP_PROP_FRAME_COUNT);
  const uint16_t width = input.get(CAP_PROP_FRAME_WIDTH);
  const uint16_t height = input.get(CAP_PROP_FRAME_HEIGHT);

  Pixel* data = new Pixel[width*height];
  uint8_t* repetitions = new uint8_t[height*width];
  for(uint64_t i = 0; i < width*height; ++i)
  repetitions[i] = 0;

  fwrite(&length, sizeof(const uint64_t), 1, output);
  fwrite(&fps, sizeof(const uint8_t), 1, output);
  fwrite(&width, sizeof(const uint16_t), 1, output);
  fwrite(&height, sizeof(const uint16_t), 1, output);

  void* status;
  uint8_t readingDone = 0;
  pthread_t cui, reading, writing;
  cuiThreadData cuiData;
  readingWorkerData readingData;
  writingWorkerData writingData;

  uint64_t readingFrameId = 0;
  uint64_t writingFrameId = 0;
  pthread_t* threads = new pthread_t[THREAD_NUM];

  Mat frame;
  readingWorkerData* readingTD = new readingWorkerData[THREAD_NUM];
  writingWorkerData* writingTD = new writingWorkerData[THREAD_NUM];
  checkingWorkerData* checkingTD = new checkingWorkerData[THREAD_NUM];
  for(uint8_t i = 0; i < THREAD_NUM; ++i)
  {
    readingTD[i].thread_id = i;
    readingTD[i].data = data;
    readingTD[i].share = (width*height) / THREAD_NUM;
    if(i == THREAD_NUM-1)
      readingTD[i].additionalShare = (width*height) - (THREAD_NUM*readingTD[i].share);
    else
      readingTD[i].additionalShare = 0;
    readingTD[i].height = height;
    readingTD[i].tollerance = tollerance;

    writingTD[i].thread_id = i;
    writingTD[i].data = data;
    writingTD[i].repetitions = repetitions;
    writingTD[i].share = (width*height) / THREAD_NUM;
    if(i == THREAD_NUM-1)
      writingTD[i].additionalShare = (width*height) - (THREAD_NUM*writingTD[i].share);
    else
      writingTD[i].additionalShare = 0;
    writingTD[i].to_dump = new uint32_t[writingTD[i].share + writingTD[i].additionalShare];
    writingTD[i].height = height;

    checkingTD[i].share = (width*height) / THREAD_NUM;
    if(i == THREAD_NUM-1)
      checkingTD[i].additionalShare = (width*height) - (THREAD_NUM*checkingTD[i].share);
    else
      checkingTD[i].additionalShare = 0;
    checkingTD[i].thread_id = i;
    checkingTD[i].data = data;
    checkingTD[i].repetitions = repetitions;
    checkingTD[i].height = height;
  }

  cuiData.readingFramesDone = &readingFrameId;
  cuiData.writingFramesDone = &writingFrameId;
  cuiData.framesTBD = length;

  pthread_create(&cui, NULL, cuiWorker, (void*)&cuiData);

  for(readingFrameId = 0; readingFrameId < length; ++readingFrameId)
  {
    //READING
    input >> frame;

    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      readingTD[i].frame = frame;
      pthread_create(&threads[i], NULL, readingWorker, (void*)&readingTD[i]);
    }
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      pthread_join(threads[i], &status);
    }

    //WRITING
    while(writingFrameId <= readingFrameId)
    {
      //CHECKING IF ABLE TO WRITE FRAME
      uint8_t checksFine = 0;
      for(uint8_t i = 0; i < THREAD_NUM; ++i)
      {
        checkingTD[i].fine = 1;
        checkingTD[i].readingDone = (uint8_t)(readingFrameId == length-1);
        pthread_create(&threads[i], NULL, checkingWorker, (void*)&checkingTD[i]);
      }
      for(uint8_t i = 0; i < THREAD_NUM; ++i)
      {
        pthread_join(threads[i], &status);
        checksFine += checkingTD[i].fine;
      }
      if(checksFine < THREAD_NUM)
        break;

      //WRITING
      for(uint8_t i = 0; i < THREAD_NUM; ++i)
      {
        writingTD[i].to_dump_size = 0;
        pthread_create(&threads[i], NULL, writingWorker, (void*)&writingTD[i]);
      }
      for(uint8_t i = 0; i < THREAD_NUM; ++i)
      {
        pthread_join(threads[i], &status);
        if(writingTD[i].to_dump_size > 0)
          fwrite(writingTD[i].to_dump, sizeof(uint32_t), writingTD[i].to_dump_size, output);
      }

      ++writingFrameId;
    }
  }

  pthread_join(cui, &status);

  input.release();
  fclose(output);

  for(uint8_t i = 0; i < THREAD_NUM; ++i)
    delete[] writingTD[i].to_dump;
  delete[] threads;
  delete[] readingTD;
  delete[] writingTD;
  delete[] repetitions;
  delete[] data;
  return 0;
}
