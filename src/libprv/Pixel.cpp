#include "RGB.hpp"
#include "Pixel.hpp"

Pixel::Record::Record(RGB initialValue)
{
  sum.R = initialValue.R;
  sum.G = initialValue.G;
  sum.B = initialValue.B;
}

RGB Pixel::Record::getValue()
{
  return RGB(
    sum.R / (repetitions + 1),
    sum.G / (repetitions + 1),
    sum.B / (repetitions + 1)
  );
}

uint8_t Pixel::Record::getRepetitions()
{
  return repetitions;
}

void Pixel::Record::addValue(RGB x)
{
  sum.R += x.R;
  sum.G += x.G;
  sum.B += x.B;
  repetitions += 1;
}

bool Pixel::Record::isFull()
{
  return repetitions == 255;
}

bool Pixel::isDumpable()
{
  return history.size() > 1;
}

bool Pixel::hasData()
{
  return history.size() > 0;
}

void Pixel::consumeValue(RGB x, uint8_t tollerance)
{
  if(history.size() == 0
  || history.back().isFull()
  || history.back().getValue().diffTo(x) > tollerance){
    Record r(x);
    history.push_back(r);
  }

  else
    history.back().addValue(x);
}

uint32_t Pixel::dumpValue()
{
  uint32_t x;
  RGB rgb = history.front().getValue();
  x = history.front().getRepetitions();
  x <<= 8;
  x += rgb.R;
  x <<= 8;
  x += rgb.G;
  x <<= 8;
  x += rgb.B;

  history.erase(history.begin());

  return x;
}
