#ifndef WRITING_HPP
#define WRITING_HPP

#define WORKER_SLEEP_TIME 10000

struct writingWorkerData
{
  uint8_t thread_id;
  Pixel* data;
  uint8_t* repetitions;
  uint64_t share;
  uint64_t additionalShare;
  uint16_t height;
  uint32_t* to_dump;
  uint64_t to_dump_size;
};

void* writingWorker(void* threadarg);

#endif
