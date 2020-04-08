#ifndef CUI_H
#define CUI_H

#define SLEEP_TIME_MS 42

struct cuiThreadData
{
  uintmax_t* framesDone;
  uintmax_t framesTBD;
};

uintmax_t getTerminalWidth();
void printProgress(uintmax_t done, uintmax_t all);
void* cuiWorker(void* threadarg);

#endif
