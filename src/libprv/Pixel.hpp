#ifndef PIXEL_HPP
#define PIXEL_HPP

#include <bits/stdc++.h>
#include "RGB.hpp"

using namespace std;

class Pixel
{
  private:
    class Record
    {
      private:
        uint8_t repetitions = 0;
        RGB sum;
      public:
        Record(RGB initialValue);
        RGB getValue();
        uint8_t getRepetitions();
        void addValue(RGB x);
        bool isFull();
    };
    vector<Record> history;

  public:
    bool isDumpable();
    void consumeValue(RGB x, uint8_t tollerance);
    uint32_t dumpValue();
    bool hasData();
};

#endif
