#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "cui.hpp"

using namespace std;

uintmax_t getTerminalWidth()
{
  struct winsize size;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
  return size.ws_col;
}

void printProgress(uintmax_t done, uintmax_t all)
{
  const uintmax_t PROGRESS_BAR_WIDTH = getTerminalWidth() - 7;
  const uintmax_t greenBlocks = (done*PROGRESS_BAR_WIDTH) / all;
  const uintmax_t percentDone = (done*100) / all;

  if(percentDone < 100)
    fprintf(stderr, " ");
  if(percentDone < 10)
    fprintf(stderr, " ");
  fprintf(stderr, "\033[1;37m %lu%% ", percentDone);

  fprintf(stderr, "\033[0;42m"); //change background to green
  for(uintmax_t i = 0; i < greenBlocks; ++i)
  {
    fprintf(stderr, " ");
  }
  fprintf(stderr, "\033[0;47m"); //change background to white
  for(uintmax_t i = greenBlocks; i < PROGRESS_BAR_WIDTH; ++i)
  {
    fprintf(stderr, " ");
  }
  fprintf(stderr, "\033[0m"); //go back to default colors

  if(percentDone == 100)
    fprintf(stderr, "\n\n");
  else
    fprintf(stderr, "\r");
}

void* cuiWorker(void* threadarg)
{
   cuiThreadData* data;
   data = (cuiThreadData*) threadarg;

   while(*data->framesDone < data->framesTBD)
   {
     usleep(SLEEP_TIME_MS * 1000);
     printProgress(*data->framesDone, data->framesTBD);
   }

   pthread_exit(NULL);
}
