#ifndef READING_HPP
#define READING_HPP

struct readingWorkerData
{
  uint64_t share;
  uint64_t additionalShare;
  uint8_t thread_id;
  cv::Mat frame;
  Pixel* data;
  uint16_t height;
  uint8_t tollerance;
};

void* readingWorker(void* threadarg);

#endif
