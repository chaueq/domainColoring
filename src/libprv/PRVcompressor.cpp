#include "PRVcompressor.hpp"

using namespace std;
using namespace cv;

PRVcompressor::PRVcompressor(const char* outputPath, uint8_t tollerance, uint8_t fps, uint64_t length, uint16_t width, uint16_t height)
{
  this->THREAD_NUM = sysconf(_SC_NPROCESSORS_ONLN);
  this->tollerance = tollerance;

  this->output = fopen(outputPath, "w");
  this->fps = fps;
  this->length = length;
  this->width = width;
  this->height = height;

  this->data = new Pixel[width*height];
  this->repetitions = new uint8_t[height*width];
  for(uint64_t i = 0; i < width*height; ++i)
    this->repetitions[i] = 0;

  fwrite(&length, sizeof(const uint64_t), 1, this->output);
  fwrite(&fps, sizeof(const uint8_t), 1, this->output);
  fwrite(&width, sizeof(const uint16_t), 1, this->output);
  fwrite(&height, sizeof(const uint16_t), 1, this->output);

  this->readingDone = 0;

  this->readingFrameId = 0;
  this->writingFrameId = 0;
  this->threads = new pthread_t[this->THREAD_NUM];

  this->readingTD = new readingWorkerData[this->THREAD_NUM];
  this->writingTD = new writingWorkerData[this->THREAD_NUM];
  this->checkingTD = new checkingWorkerData[this->THREAD_NUM];

  for(uint8_t i = 0; i < THREAD_NUM; ++i)
  {
    this->readingTD[i].thread_id = i;
    this->readingTD[i].data = this->data;
    this->readingTD[i].share = (width*height) / this->THREAD_NUM;
    if(i == this->THREAD_NUM-1)
      this->readingTD[i].additionalShare = (width*height) - (this->THREAD_NUM*this->readingTD[i].share);
    else
      this->readingTD[i].additionalShare = 0;
    this->readingTD[i].height = height;
    this->readingTD[i].tollerance = tollerance;

    this->writingTD[i].thread_id = i;
    this->writingTD[i].data = this->data;
    this->writingTD[i].repetitions = this->repetitions;
    this->writingTD[i].share = (width*height) / this->THREAD_NUM;
    if(i == this->THREAD_NUM-1)
      this->writingTD[i].additionalShare = (width*height) - (this->THREAD_NUM*this->writingTD[i].share);
    else
      this->writingTD[i].additionalShare = 0;
    this->writingTD[i].to_dump = new uint32_t[this->writingTD[i].share + this->writingTD[i].additionalShare];
    this->writingTD[i].height = height;

    this->checkingTD[i].share = (width*height) / this->THREAD_NUM;
    if(i == this->THREAD_NUM-1)
      this->checkingTD[i].additionalShare = (width*height) - (this->THREAD_NUM*this->checkingTD[i].share);
    else
      this->checkingTD[i].additionalShare = 0;
    this->checkingTD[i].thread_id = i;
    this->checkingTD[i].data = this->data;
    this->checkingTD[i].repetitions = this->repetitions;
    this->checkingTD[i].height = height;
  }
}

void PRVcompressor::upload(Mat frame)
{
  //READING
  {
    for(uint8_t i = 0; i < this->THREAD_NUM; ++i)
    {
      this->readingTD[i].frame = frame;
      pthread_create(&this->threads[i], NULL, readingWorker, (void*)&this->readingTD[i]);
    }
    for(uint8_t i = 0; i < this->THREAD_NUM; ++i)
    {
      pthread_join(this->threads[i], &this->status);
    }

    ++this->readingFrameId;
  }

  while(this->writingFrameId <= this->readingFrameId)
  {
    //CHECKING IF ABLE TO WRITE FRAME
    uint8_t checksFine = 0;
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      this->checkingTD[i].fine = 1;
      this->checkingTD[i].readingDone = (uint8_t)(this->readingFrameId == this->length-1);
      pthread_create(&this->threads[i], NULL, checkingWorker, (void*)&this->checkingTD[i]);
    }
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      pthread_join(this->threads[i], &this->status);
      checksFine += this->checkingTD[i].fine;
    }
    if(checksFine < this->THREAD_NUM)
      break;

    //WRITING
    for(uint8_t i = 0; i < this->THREAD_NUM; ++i)
    {
      this->writingTD[i].to_dump_size = 0;
      pthread_create(&this->threads[i], NULL, writingWorker, (void*)&this->writingTD[i]);
    }
    for(uint8_t i = 0; i < THREAD_NUM; ++i)
    {
      pthread_join(this->threads[i], &this->status);
      if(this->writingTD[i].to_dump_size > 0)
        fwrite(this->writingTD[i].to_dump, sizeof(uint32_t), this->writingTD[i].to_dump_size, this->output);
    }

    ++this->writingFrameId;
  }
}

PRVcompressor::~PRVcompressor()
{
  fclose(output);
  for(uint8_t i = 0; i < this->THREAD_NUM; ++i)
    delete[] this->writingTD[i].to_dump;
  delete[] this->threads;
  delete[] this->readingTD;
  delete[] this->writingTD;
  delete[] this->repetitions;
  delete[] this->data;
}
