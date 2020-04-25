#ifndef CHECKING_HPP
#define CHECKING_HPP

struct checkingWorkerData
{
  uint64_t share;
  uint64_t additionalShare;
  uint8_t thread_id;
  uint8_t fine;
  Pixel* data;
  uint8_t* repetitions;
  uint8_t readingDone;
  uint16_t height;
};

void* checkingWorker(void* threadarg);

#endif
