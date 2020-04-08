#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <stdio.h>
#include <sys/resource.h>
#include "RGB.hpp"
#include "Pixel.hpp"
#include "cui.hpp"

using namespace std;
using namespace cv;

struct threadData
{
  uint8_t num_threads;
  uint8_t thread_id;
  Pixel** data;
  uint8_t** repetitions;
  Mat frame;
  uint64_t to_dump_size;
  uint32_t* to_dump;
  uint64_t share;
  uint64_t additionalShare;
  uint8_t tollerance;
  uint16_t width, height;
};

void* loadingThread(void* threadarg);
void* writingThread(void* threadarg);

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
  const uint8_t fps = input.get(CAP_PROP_FPS);
  const uint64_t length = input.get(CAP_PROP_FRAME_COUNT);
  const uint16_t width = input.get(CAP_PROP_FRAME_WIDTH);
  const uint16_t height = input.get(CAP_PROP_FRAME_HEIGHT);
  FILE* output = fopen(outputPath, "w");
  uint64_t frameId;
  Mat frame;
  Pixel** data = new Pixel*[width];
  for(uint16_t i = 0; i < width; ++i)
  {
    data[i] = new Pixel[height];
  }
  uint8_t** repetitions = new uint8_t*[width];
  for(uint16_t x = 0; x < width; ++x)
  {
    repetitions[x] = new uint8_t[height];
    for(uint16_t y = 0; y < height; ++y)
      repetitions[x][y] = 0;
  }
  void* status;
  pthread_t* threads = new pthread_t[THREAD_NUM];
  threadData* td = new threadData[THREAD_NUM];
  for(uint8_t i = 0; i < THREAD_NUM; ++i)
  {
    td[i].num_threads = THREAD_NUM;
    td[i].thread_id = i;
    td[i].data = data;
    td[i].repetitions = repetitions;
    td[i].share = (width*height) / THREAD_NUM;
    if(i == THREAD_NUM-1)
      td[i].additionalShare = (width*height) - (THREAD_NUM*td[i].share);
    else
      td[i].additionalShare = 0;
    td[i].to_dump = new uint32_t[td[i].share + td[i].additionalShare];
    td[i].tollerance = tollerance;
    td[i].width = width;
    td[i].height = height;
  }
  pthread_t cuiThread;
  cuiThreadData cuiTD;
  uintmax_t progress = 0;
  cuiTD.framesDone = &progress;
  cuiTD.framesTBD = 2*length;
  pthread_create(&cuiThread, NULL, cuiWorker, (void*)&cuiTD);

  //LOADING
  for(frameId = 0; frameId < length; ++frameId)
  {
    input >> frame;

    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      td[i].frame = frame;
      pthread_create(&threads[i], NULL, loadingThread, (void*)&td[i]);
    }
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      pthread_join(threads[i], &status);
    }

    ++progress;
  }

  //WRITING
  fwrite(&length, sizeof(const uint64_t), 1, output);
  fwrite(&fps, sizeof(const uint8_t), 1, output);
  fwrite(&width, sizeof(const uint16_t), 1, output);
  fwrite(&height, sizeof(const uint16_t), 1, output);
  for(uint64_t frameId = 0; frameId < length; ++frameId)
  {
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      td[i].to_dump_size = 0;
      pthread_create(&threads[i], NULL, writingThread, (void*)&td[i]);
    }

    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      pthread_join(threads[i], &status);
      if(td[i].to_dump_size > 0)
        fwrite(td[i].to_dump, sizeof(uint32_t), td[i].to_dump_size, output);
    }

    ++progress;
  }

  pthread_join(cuiThread, &status);
  input.release();
  fclose(output);
  delete[] threads;

  for(uint8_t i = 0; i < THREAD_NUM; ++i)
    delete[] td[i].to_dump;
  delete[] td;

  for(uint16_t i = 0; i < width; ++i)
  {
    delete[] data[i];
  }
  delete[] data;
  return 0;
}

void* loadingThread(void* threadarg)
{
  threadData* td = (threadData*) threadarg;

  for(uint64_t i = 0; i < td->share + td->additionalShare; ++i)
  {
    const uint16_t x = (i + (td->share*td->thread_id)) / (td->height);
    const uint16_t y = (i + (td->share*td->thread_id)) % (td->height);
    const RGB value(
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 2],
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 1],
      td->frame.data[(3*(i + (td->share*td->thread_id))) + 0]
    );

    td->data[x][y].consumeValue(value, td->tollerance);
  }

  pthread_exit(NULL);
}

void* writingThread(void* threadarg)
{
  threadData* td = (threadData*) threadarg;

  for(uint64_t i = 0; i < td->share + td->additionalShare; ++i)
  {
    const uint16_t x = (i + (td->share*td->thread_id)) / (td->height);
    const uint16_t y = (i + (td->share*td->thread_id)) % (td->height);

    if(td->repetitions[x][y] == 0){
      td->to_dump[td->to_dump_size] = td->data[x][y].dumpValue();
      td->repetitions[x][y] = (td->to_dump[td->to_dump_size] >> 24) % 256;
      td->to_dump_size += 1;
    }

    else
    {
      td->repetitions[x][y] -= 1;
    }
  }

  pthread_exit(NULL);
}
