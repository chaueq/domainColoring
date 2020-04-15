#ifndef CUI_H
#define CUI_H

#define SLEEP_TIME_MS 100

struct cuiThreadData
{
  uint64_t* readingFramesDone;
  uint64_t* writingFramesDone;
  uint64_t framesTBD;
};

uintmax_t getTerminalWidth();
void printProgress(uintmax_t done, uintmax_t all);
void* cuiWorker(void* threadarg);

#endif
