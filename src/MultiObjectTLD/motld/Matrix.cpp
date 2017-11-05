#include "Matrix.hpp"
/**************************************************************************************************
 * IMPLEMENTATION                                                                                 *
 **************************************************************************************************/ 

Matrix::Matrix(const Matrix& copyFrom)
  : ivWidth(copyFrom.ivWidth), ivHeight(copyFrom.ivHeight)
{
  if (copyFrom.ivData == 0) ivData = 0;
  else {
    int wholeSize = ivWidth*ivHeight;
    ivData = new float[wholeSize];
    memcpy(ivData, copyFrom.ivData, wholeSize * sizeof(float));
    //for (register int i = 0; i < wholeSize; i++)
    //  ivData[i] = copyFrom.ivData[i];
  }
}

Matrix::Matrix(const int width, const int height, const float value)
  : ivWidth(width), ivHeight(height)
{
  ivData = new float[width*height];
  fill(value);
}

Matrix::~Matrix()
{
  delete [] ivData;
}

void Matrix::copyFromCharArray(unsigned char * source)
{
  delete [] ivData;
  int wholeSize = ivWidth*ivHeight;
  ivData = new float[wholeSize];
  for (register int i = 0; i < wholeSize; ++i)
    ivData[i] = (float)source[i];
}

void Matrix::copyFromFloatArray(const float * const source, int srcwidth, int srcheight, 
                                              int x, int y, int width, int height)
{
  delete [] ivData;
  ivWidth = width; ivHeight = height;
  ivData = new float[width*height];
  #pragma omp parallel for
  for (int dy = 0; dy < height; ++dy)
    memcpy(ivData + dy * width, source + (y + dy) * srcwidth + x, width * sizeof(float));
}

void Matrix::copyFromFloatArray(float * source, int srcwidth, int width, int height)
{
  delete [] ivData;
  ivWidth = width; ivHeight = height;
  ivData = new float[width*height];
  for (int dy = 0; dy < height; ++dy)
    memcpy(ivData + dy * width, source + dy * srcwidth, width * sizeof(float));
}

void Matrix::fromRGB(const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix) 
{
  //delete [] ivData;
  int wholeSize = ivWidth*ivHeight;
  //ivData = new float[wholeSize];
  float * rData = rMatrix.data();
  float * gData = gMatrix.data();
  float * bData = bMatrix.data();
  for (int i = 0; i < wholeSize; ++i)
    ivData[i] = (rData[i] + gData[i] + bData[i]) * (1.0/3.0);
}

void Matrix::fromRGB(unsigned char * source)
{
  //delete [] ivData;
  int wholeSize = ivWidth*ivHeight;
  //ivData = new float[wholeSize];
  unsigned char * green = source + wholeSize;
  unsigned char * blue = green + wholeSize;
  for (int i = 0; i < wholeSize; ++i)
    ivData[i] = ((float)source[i] + (float)green[i] + (float)blue[i]) * (1.0/3.0);
}

void Matrix::derivativeX(Matrix& result) const
{
  result.setSize(ivWidth, ivHeight);
  for(int y = 0; y < ivHeight; ++y)
  {
    result(0,y) = ivData[1 + y*ivWidth] - ivData[y*ivWidth];
     for(int x = 1; x < ivWidth-1; ++x)
       result(x,y) = (ivData[x+1 +y*ivWidth] - ivData[x-1 +y*ivWidth]); // * 0.5;   
    result(ivWidth-1,y) = ivData[ivWidth-1 + y*ivWidth] - ivData[ivWidth-2 + y*ivWidth];   
  }
}

void Matrix::derivativeY(Matrix& result) const
{
  result.setSize(ivWidth, ivHeight);
  for(int x = 0; x < ivWidth; ++x)
  {
    result(x,0) = ivData[x + ivWidth] - ivData[x];
    result(x,ivHeight-1) = ivData[x + (ivHeight-1)*ivWidth] - ivData[x + (ivHeight-2)*ivWidth];    
  }
  for(int y = 1; y < ivHeight-1; ++y)
    for(int x = 0; x < ivWidth; ++x)
       result(x,y) = (ivData[x + (y+1)*ivWidth] - ivData[x + (y-1)*ivWidth]); // * 0.5;   
}

/// @details Applied filter: [-3,0,3; -10,0,10; -3,0,3] = [-1,0,1] x [3;10;3]
void Matrix::scharrDerivativeX(Matrix& result) const
{
  if(ivWidth * ivHeight == 0)return;
  Matrix tmp;
  this->derivativeX(tmp); //apply [-1,0,1]
  result.setSize(ivWidth, ivHeight);
  //apply [3;10;3]
  for(int x=0; x<ivWidth; ++x)
  {
    result(x,0) = 13 * tmp(x,0) + 3 * tmp(x,1);
    result(x,ivHeight-1) = 13 * tmp(x,ivHeight-1) + 3 * tmp(x,ivHeight-2);   
  }
  for(int y = 1; y < ivHeight-1; ++y)
    for(int x = 0; x < ivWidth; ++x)
       result(x,y) = 3 * (tmp(x,y-1) + tmp(x,y+1)) + 10 * tmp(x,y);   

}

/// @see scharrDerivativeX()
void Matrix::scharrDerivativeY(Matrix& result) const
{
  if(ivWidth * ivHeight == 0)return;
  Matrix tmp;
  this->derivativeY(tmp);
  result.setSize(ivWidth, ivHeight);
  for(int y = 0; y < ivHeight; ++y)
  {
    result(0,y) = 13 * tmp(0,y) + 3 * tmp(1,y);
     for(int x = 1; x < ivWidth-1; ++x)
       result(x,y) = 3 * (tmp(x-1,y) + tmp(x+1,y)) + 10 * tmp(x,y);   
    result(ivWidth-1,y) = 13 * tmp(ivWidth-1,y) + 3 * tmp(ivWidth-2,y);   
  }
}

/// @details Applied filter: [-1,0,1; -2,0,2; -1,0,1] = [-1,0,1] x [1;2;1]
void Matrix::sobelDerivativeX(Matrix& result) const
{
  if(ivWidth * ivHeight == 0)return;
  Matrix tmp;
  this->derivativeX(tmp); //apply [-1,0,1]
  result.setSize(ivWidth, ivHeight);
  //apply [1;2;1]
  for(int x=0; x<ivWidth; ++x)
  {
    result(x,0) = 3 * tmp(x,0) + 1 * tmp(x,1);
    result(x,ivHeight-1) = 3 * tmp(x,ivHeight-1) + 1 * tmp(x,ivHeight-2);   
     for(int y = 1; y < ivHeight-1; ++y)
       result(x,y) = 1 * (tmp(x,y-1) + tmp(x,y+1)) + 2 * tmp(x,y);   
  }
}

/// @see sobelDerivativeX()
void Matrix::sobelDerivativeY(Matrix& result) const
{
  if(ivWidth * ivHeight == 0)return;
  Matrix tmp;
  this->derivativeY(tmp);
  result.setSize(ivWidth, ivHeight);
  for(int y=0; y<ivHeight; ++y)
  {
    result(0,y) = 3 * tmp(0,y) + 1 * tmp(1,y);
    result(ivWidth-1,y) = 3 * tmp(ivWidth-1,y) + 1 * tmp(ivWidth-2,y);   
     for(int x = 1; x < ivWidth-1; ++x)
       result(x,y) = 1 * (tmp(x-1,y) + tmp(x+1,y)) + 2 * tmp(x,y);   
  }
}

void Matrix::gaussianSmooth(const float sigma, const int filterSize)
{
  Matrix temp(ivWidth, ivHeight, 0);
  int fSize = filterSize > 0 ? filterSize : (sigma*6 + 1);
  //force to be odd
  if (!(fSize%2))
    fSize++;
  // compute gaussian weights
  float* weights = new float[fSize];
  float sumWeights = 0;
  for (int i = 0; i < fSize; i++)
  {
    float x = (i - (fSize>>1));
    weights[i] =  1.0 / (sqrt(2*PI) * sigma) * exp(-x*x / (2*sigma*sigma));
    sumWeights += weights[i];
  }
  // normalize weights
  for (int i = 0; i < fSize; i++)
    weights[i] *= 1.0 / sumWeights;
    
  // apply filter in x-direction  
  for (int x = 0; x < ivWidth; x++)
    for (int i = 0; i < fSize; i++)
    {
      int xtemp = x + i - (fSize>>1);
      xtemp = xtemp < 0 ? 0 : (xtemp >= ivWidth ? ivWidth-1 : xtemp);
      for (int y = 0; y < ivHeight; y++)
        temp(x,y) += ivData[xtemp + y*ivWidth] * weights[i];
    }      
  // apply filter in y-direction  
  fill(0);
  for (int y = 0; y < ivHeight; y++)
    for (int i = 0; i < fSize; i++)
    {
      int ytemp = y + i - (fSize>>1);
      ytemp = ytemp < 0 ? 0 : (ytemp >= ivHeight ? ivHeight-1 : ytemp);
      for (int x = 0; x < ivWidth; x++)
        ivData[x + y*ivWidth] += temp(x, ytemp) * weights[i];
    }
  delete [] weights;
}

/// @details Applies the filter [1/4 1/2 1/4]^2
//maybe use [1/16 1/4 3/8 1/4 1/16]^2 instead
void Matrix::halfSizeImage(Matrix& result) const 
{  
  //downsample in x-direction
  Matrix temp((ivWidth+1)>>1, ivHeight);
  for (int y = 0; y < ivHeight; ++y)
  {
    temp(0,y) = 0.75 * ivData[0 + y*ivWidth] + 0.25 * ivData[1 + y*ivWidth];
    if (ivWidth%2) //odd
      temp(ivWidth>>1,y) = 0.75 * ivData[ivWidth-1 + y*ivWidth] + 0.25 * ivData[ivWidth-2 + y*ivWidth];
     for (int x = 1; x < (ivWidth>>1); ++x)
       temp(x,y) = 0.5 * ivData[(x<<1) + y*ivWidth] + 0.25 * (ivData[(x<<1)-1 + y*ivWidth] + ivData[(x<<1)+1 + y*ivWidth]);
  }  
  //downsample in y-direction
  result.setSize((ivWidth+1)>>1, (ivHeight+1)>>1);  
  for (int x = 0; x < result.ivWidth; ++x)
  {
    result(x,0) = 0.75 * temp(x,0) + 0.25 * temp(x,1);
    if (ivHeight%2) //odd
      result(x,ivHeight>>1) = 0.75 * temp(x,ivHeight-1) + 0.25 * temp(x,ivHeight-2);
     for (int y = 1; y < (ivHeight>>1); ++y)
       result(x,y) = 0.5 * temp(x,y<<1) + 0.25 * (temp(x,(y<<1)-1) + temp(x,(y<<1)+1));
  }
}

void Matrix::writeToPGM(const char *filename) const
{
  FILE *aStream;
  aStream = fopen(filename,"wb");
  // write header
  char line[60];
  sprintf(line,"P5\n%d %d\n255\n",ivWidth,ivHeight);
  fwrite(line,strlen(line),1,aStream);
  // write data
  for (int i = 0; i < ivWidth*ivHeight; i++) {
    char dummy = (char)ivData[i];
    fwrite(&dummy,1,1,aStream);
  }
  fclose(aStream);
}

Matrix Matrix::getRectSubPix(float centerx, float centery, int width, int height) const
{
  Matrix result(width, height);
  float cx = centerx - (width-1)*0.5f, cy = centery - (height-1)*0.5f;
  int srcx = floor(cx), srcy = floor(cy);
  float a = cx - srcx, b = cy - srcy,
    a11 = (1.f-a)*(1.f-b),
    a12 = a*(1.f-b),
    a21 = (1.f-a)*b,
    a22 = a*b;
  if(srcx >= 0 && srcy >= 0 && srcx+width < ivWidth && srcy+height < ivHeight)
  { // patch is completely inside the image
    for(int y=0; y<height; ++y)
    {
      for(int x=0; x<width; ++x)
      {
        result(x,y) = a11*operator()(srcx+x,srcy+y)
                    + a12*operator()(srcx+x+1,srcy+y)
                    + a21*operator()(srcx+x,srcy+y+1)
                    + a22*operator()(srcx+x+1,srcy+y+1);    
      }
    }
  }else{ 
    float avgValue = this->avg();
    for(int y=0; y<height; ++y)
    {
      for(int x=0; x<width; ++x)
      {
        if(srcx+x<0 || srcx+x+1>=ivWidth || srcy+y<0 || srcy+y+1>=ivHeight)
          result(x,y) = avgValue;
        else
          result(x,y) = a11*operator()(srcx+x,srcy+y)
                      + a12*operator()(srcx+x+1,srcy+y)
                      + a21*operator()(srcx+x,srcy+y+1)
                      + a22*operator()(srcx+x+1,srcy+y+1);  
        /*
        // copy from border (not very efficient)  
        result(x,y) = a11*operator()(MAX(0,MIN(ivWidth-1,srcx+x)),MAX(0,MIN(ivHeight-1,srcy+y)))
                    + a12*operator()(MAX(0,MIN(ivWidth-1,srcx+x+1)),MAX(0,MIN(ivHeight-1,srcy+y)))
                    + a21*operator()(MAX(0,MIN(ivWidth-1,srcx+x)),MAX(0,MIN(ivHeight-1,srcy+y+1)))
                    + a22*operator()(MAX(0,MIN(ivWidth-1,srcx+x+1)),MAX(0,MIN(ivHeight-1,srcy+y+1)));  */  
      }
    }
  }
  return result;
}

void Matrix::setSize(int width, int height)
{
  if (ivWidth == width && ivHeight == height)
    return;
  if (ivData != 0) 
    delete[] ivData;
  ivData = new float[width*height];
  ivWidth = width;
  ivHeight = height;
}

void Matrix::downsample(int newWidth, int newHeight)
{
  // Downsample in x-direction
  int aIntermedSize = newWidth*ivHeight;
  float* aIntermedData = new float[aIntermedSize];
  if (newWidth < ivWidth) {
    for (int i = 0; i < aIntermedSize; i++)
      aIntermedData[i] = 0.0;
    float factor = ((float)ivWidth)/newWidth;
    for (int y = 0; y < ivHeight; y++) {
      int aFineOffset = y*ivWidth;
      int aCoarseOffset = y*newWidth;
      int i = aFineOffset;
      int j = aCoarseOffset;
      int aLastI = aFineOffset+ivWidth;
      int aLastJ = aCoarseOffset+newWidth;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          aIntermedData[j] += part*ivData[i];
          rest -= part;
          part = 1.0;
          i++;
          if (rest <= 0.0) {
            rest = factor;
            j++;
          }
        }
        else {
          aIntermedData[j] += rest*ivData[i];
          part = 1.0-rest;
          rest = factor;
          j++;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    float* aTemp = aIntermedData;
    aIntermedData = ivData;
    ivData = aTemp;
  }
  // Downsample in y-direction
  delete[] ivData;
  int aDataSize = newWidth*newHeight;
  ivData = new float[aDataSize];
  if (newHeight < ivHeight) {
    for (int i = 0; i < aDataSize; i++)
      ivData[i] = 0.0;
    float factor = ((float)ivHeight)/newHeight;
    for (int x = 0; x < newWidth; x++) {
      int i = x;
      int j = x;
      int aLastI = ivHeight*newWidth+x;
      int aLastJ = newHeight*newWidth+x;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          ivData[j] += part*aIntermedData[i];
          rest -= part;
          part = 1.0;
          i += newWidth;
          if (rest <= 0.0) {
            rest = factor;
            j += newWidth;
          }
        }
        else {
          ivData[j] += rest*aIntermedData[i];
          part = 1.0-rest;
          rest = factor;
          j += newWidth;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    float* aTemp = ivData;
    ivData = aIntermedData;
    aIntermedData = aTemp;
  }
  // Normalize
  float aNormalization = ((float)aDataSize)/size();
  for (int i = 0; i < aDataSize; i++)
    ivData[i] *= aNormalization;
  // Adapt size of matrix
  ivWidth = newWidth;
  ivHeight = newHeight;
  delete[] aIntermedData;
}

void Matrix::downsampleBilinear(int newWidth, int newHeight)
{
  int newSize = newWidth*newHeight;
  float* newData = new float[newSize];
  float factorX = ((float)ivWidth)/newWidth;
  float factorY = ((float)ivHeight)/newHeight;
  for (int y = 0; y < newHeight; y++)
    for (int x = 0; x < newWidth; x++) {
      float ax = (x+0.5)*factorX-0.5;
      float ay = (y+0.5)*factorY-0.5;
      if (ax < 0) ax = 0.0;
      if (ay < 0) ay = 0.0;
      int x1 = (int)ax;
      int y1 = (int)ay;
      int x2 = x1+1;
      int y2 = y1+1;
      float alphaX = ax-x1;
      float alphaY = ay-y1;
      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 >= ivWidth) x2 = ivWidth-1;
      if (y2 >= ivHeight) y2 = ivHeight-1;
      float a = (1.0-alphaX)*ivData[x1+y1*ivWidth]+alphaX*ivData[x2+y1*ivWidth];
      float b = (1.0-alphaX)*ivData[x1+y2*ivWidth]+alphaX*ivData[x2+y2*ivWidth];
      newData[x+y*newWidth] = (1.0-alphaY)*a+alphaY*b;
    }
  delete[] ivData;
  ivData = newData;
  ivWidth = newWidth;
  ivHeight = newHeight;
}

void Matrix::upsample(int newWidth, int newHeight)
{
  // Upsample in x-direction
  int aIntermedSize = newWidth*ivHeight;
  float* aIntermedData = new float[aIntermedSize];
  if (newWidth > ivWidth) {
    for (int i = 0; i < aIntermedSize; i++)
      aIntermedData[i] = 0.0;
    float factor = ((float)newWidth)/ivWidth;
    for (int y = 0; y < ivHeight; y++) {
      int aFineOffset = y*newWidth;
      int aCoarseOffset = y*ivWidth;
      int i = aCoarseOffset;
      int j = aFineOffset;
      int aLastI = aCoarseOffset+ivWidth;
      int aLastJ = aFineOffset+newWidth;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          aIntermedData[j] += part*ivData[i];
          rest -= part;
          part = 1.0;
          j++;
          if (rest <= 0.0) {
            rest = factor;
            i++;
          }
        }
        else {
          aIntermedData[j] += rest*ivData[i];
          part = 1.0-rest;
          rest = factor;
          i++;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    float* aTemp = aIntermedData;
    aIntermedData = ivData;
    ivData = aTemp;
  }
  // Upsample in y-direction
  delete[] ivData;
  int aDataSize = newWidth*newHeight;
  ivData = new float[aDataSize];
  if (newHeight > ivHeight) {
    for (int i = 0; i < aDataSize; i++)
      ivData[i] = 0.0;
    float factor = ((float)newHeight)/ivHeight;
    for (int x = 0; x < newWidth; x++) {
      int i = x;
      int j = x;
      int aLastI = ivHeight*newWidth;
      int aLastJ = newHeight*newWidth;
      float rest = factor;
      float part = 1.0;
      do {
        if (rest > 1.0) {
          ivData[j] += part*aIntermedData[i];
          rest -= part;
          part = 1.0;
          j += newWidth;
          if (rest <= 0.0) {
            rest = factor;
            i += newWidth;
          }
        }
        else {
          ivData[j] += rest*aIntermedData[i];
          part = 1.0-rest;
          rest = factor;
          i += newWidth;
        }
      }
      while (i < aLastI && j < aLastJ);
    }
  }
  else {
    float* aTemp = ivData;
    ivData = aIntermedData;
    aIntermedData = aTemp;
  }
  // Adapt size of matrix
  ivWidth = newWidth;
  ivHeight = newHeight;
  delete[] aIntermedData;
}

void Matrix::upsampleBilinear(int newWidth, int newHeight)
{
  int newSize = newWidth*newHeight;
  float* newData = new float[newSize];
  float factorX = (float)(ivWidth)/(newWidth);
  float factorY = (float)(ivHeight)/(newHeight);
  for (int y = 0; y < newHeight; y++)
    for (int x = 0; x < newWidth; x++) {
      float ax = (x+0.5)*factorX-0.5;
      float ay = (y+0.5)*factorY-0.5;
      if (ax < 0) ax = 0.0;
      if (ay < 0) ay = 0.0;
      int x1 = (int)ax;
      int y1 = (int)ay;
      int x2 = x1+1;
      int y2 = y1+1;
      float alphaX = ax-x1;
      float alphaY = ay-y1;
      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 >= ivWidth) x2 = ivWidth-1;
      if (y2 >= ivHeight) y2 = ivHeight-1;
      float a = (1.0-alphaX)*ivData[x1+y1*ivWidth]+alphaX*ivData[x2+y1*ivWidth];
      float b = (1.0-alphaX)*ivData[x1+y2*ivWidth]+alphaX*ivData[x2+y2*ivWidth];
      newData[x+y*newWidth] = (1.0-alphaY)*a+alphaY*b;
    }
  delete[] ivData;
  ivData = newData;
  ivWidth = newWidth;
  ivHeight = newHeight;
}

void Matrix::rescale(int newWidth, int newHeight)
{
  if (ivWidth >= newWidth) {
    if (ivHeight >= newHeight) 
      downsample(newWidth,newHeight);
    else {
      downsample(newWidth,ivHeight);
      upsample(newWidth,newHeight);
    }
  }
  else {
    if (ivHeight >= newHeight) {
      downsample(ivWidth,newHeight);
      upsample(newWidth,newHeight);
    }
    else 
      upsample(newWidth,newHeight);
  }
}

void Matrix::fill(const float value)
{
  int wholeSize = ivWidth*ivHeight;
  for (register int i = 0; i < wholeSize; i++)
    ivData[i] = value;
}

void Matrix::cut(Matrix& result,const int x1, const int y1, const int x2, const int y2)
{
  result.ivWidth = x2-x1+1;
  result.ivHeight = y2-y1+1;
  delete[] result.ivData;
  result.ivData = new float[result.ivWidth*result.ivHeight];
  for (int y = y1; y <= y2; y++)
    for (int x = x1; x <= x2; x++)
      result(x-x1,y-y1) = operator()(x,y);
}

void Matrix::clip(float aMin, float aMax)
{
  int aSize = size();
  for (int i = 0; i < aSize; i++)
    if (ivData[i] < aMin) 
      ivData[i] = aMin;
    else if (ivData[i] > aMax) 
      ivData[i] = aMax;
}

void Matrix::inv3()
{
  if (ivWidth != ivHeight || ivWidth != 3) {
    std::cerr << "cannot invert non 3x3 matrices!" << std::endl;
    return;  
  }
  
  float a,b,c,d,e,f,g,h,k;
  a = ivData[0]; b = ivData[3]; c = ivData[6];
  d = ivData[1]; e = ivData[4]; f = ivData[7];
  g = ivData[2]; h = ivData[5]; k = ivData[8];
  
  float A = e*k - f*h;
  float B = f*g - d*k;
  float C = d*h - e*g;
  float D = c*h - b*k;
  float E = a*k - c*g;
  float F = g*b - a*h;
  float G = b*f - c*e;
  float H = c*d - a*f;
  float K = a*e - b*d;
  
  float det = a*A + b*B + c*C;
  
  ivData[0] = A/det; ivData[3] = D/det; ivData[6] = G/det;
  ivData[1] = B/det; ivData[4] = E/det; ivData[7] = H/det;
  ivData[2] = C/det; ivData[5] = F/det; ivData[8] = K/det;

}

void Matrix::drawLine(int x1, int y1, int x2, int y2, float value)
{
  // vertical line
  if (x1 == x2) 
  {
    if (x1 < 0 || x1 >= ivWidth)   
      return;
    int x = x1;
    if (y1 < y2) 
    {
      for (int y = y1; y <= y2; y++)
        if (y >= 0 && y < ivHeight) 
          ivData[y*ivWidth + x] = value;
    } else {
      for (int y = y1; y >= y2; y--)
        if (y >= 0 && y < ivHeight) 
          ivData[y*ivWidth + x] = value;
    }
    return;
  }
  // horizontal line
  if (y1 == y2) 
  {
    if (y1 < 0 || y1 >= ivHeight)
      return;
    int y = y1;
    if (x1 < x2) 
    {
      for (int x = x1; x <= x2; x++)
        if (x >= 0 && x < ivWidth)
          ivData[y*ivWidth + x] = value;
    } else {
      for (int x = x1; x >= x2; x--)
        if (x >= 0 && x < ivWidth) 
          ivData[y*ivWidth + x] = value;
    }
    return;
  }
  float m = float(y1 - y2) / float(x1 - x2);
  float invm = 1.0/m;
  if (fabs(m) > 1.0) 
  {
    if (y2 > y1) 
    {
      for (int y = y1; y <= y2; y++) 
      {
        int x = (int)(0.5 + x1 + (y-y1)*invm);
        if (x >= 0 && x < ivWidth && y >= 0 && y < ivHeight)
          ivData[y*ivWidth + x] = value;
      }
    } else {
      for (int y = y1; y >= y2; y--) 
      {
        int x = (int)(0.5 + x1 + (y-y1)*invm);
        if (x >= 0 && x < ivWidth && y >= 0 && y < ivHeight)
          ivData[y*ivWidth + x] = value;
      }
    }
  } else {
    if (x2 > x1) 
    {
      for (int x = x1; x <= x2; x++) 
      {
        int y = (int)(0.5 + y1 + (x-x1)*m);
        if (x >= 0 && x < ivWidth && y >= 0 && y < ivHeight)
          ivData[y*ivWidth + x] = value;
      }
    } else {
      for (int x = x1; x >= x2; x--)
      {
        int y = (int)(0.5 + y1 + (x-x1)*m);
        if (x >= 0 && x < ivWidth && y >= 0 && y < ivHeight)
          ivData[y*ivWidth + x] = value;
      }
    }
  }
}

void Matrix::drawCross(int x, int y, int value, int crossSize)
{
  if(x > crossSize && y > crossSize && x < ivWidth-crossSize && y < ivHeight-crossSize)
    for (int dx = -crossSize; dx <= crossSize; ++dx)
    {
      int oy = (y+dx)*ivWidth;
      ivData[oy+x+dx] = value;
      ivData[oy+x-dx] = value;
    }
}

void Matrix::drawBox(ObjectBox b, int value)
{
  int x1 = round(b.x), x2 = round(b.x + b.width),
      y1 = round(b.y), y2 = round(b.y + b.height);
  if(x2 < 0 || y2 < 0 || x1 >= ivWidth || y1 >= ivHeight)
    return;
  if(y1 >= 0)
    for(int i = y1 * ivWidth + std::max(0, x1);
            i < y1 * ivWidth + std::min(ivWidth, x2); ++i)
      ivData[i] = value;
  if(y2 < ivHeight)
    for(int i = y2 * ivWidth + std::max(0, x1);
            i < y2 * ivWidth + std::min(ivWidth, x2); ++i)
      ivData[i] = value;
  if(x1 >= 0)
    for(int i = std::max(0, y1) * ivWidth + x1;
            i < std::min(ivHeight, y2) * ivWidth + x1; i += ivWidth)
      ivData[i] = value;
  if(x2 < ivWidth)
    for(int i = std::max(0, y1) * ivWidth + x2;
            i < std::min(ivHeight, y2) * ivWidth + x2; i += ivWidth)
      ivData[i] = value;
}

void Matrix::drawDashedBox(ObjectBox b, int value, int dashLength, bool dotted)
{
  int x1 = round(b.x), x2 = round(b.x + b.width),
      y1 = round(b.y), y2 = round(b.y + b.height);
  if(x2 < 0 || y2 < 0 || x1 >= ivWidth || y1 >= ivHeight)
    return;
  for (int dx = 0; dx < b.width; ++dx)
  {
    int i1 = y1 * ivWidth + x1 + dx;
    int i2 = y2 * ivWidth + x1 + dx;
    if (0 <= i1 && i1 < ivWidth * ivHeight && ((dx%dashLength)>0)^dotted)
      ivData[i1] = value;
    if (0 <= i2 && i2 < ivWidth * ivHeight && ((dx%dashLength)>0)^dotted)
      ivData[i2] = value;
  }
  for (int dy = 0; dy < b.height; ++dy)
  {
    int i1 = (y1 + dy) * ivWidth + x1;
    int i2 = (y1 + dy) * ivWidth + x2;
    if (0 <= i1 && i1 < ivWidth * ivHeight && ((dy%dashLength)>0)^dotted)
      ivData[i1] = value;
    if (0 <= i2 && i2 < ivWidth * ivHeight && ((dy%dashLength)>0)^dotted)
      ivData[i2] = value;
  }
}

void Matrix::drawPatch(const Matrix& b, int x, int y, float avg)
{
  for (int dx = 0; dx < b.ivWidth; ++dx)
    for (int dy = 0; dy < b.ivHeight; ++ dy)
      (*this)(x+dx,y+dy) = b(dx,dy) + avg;
}

void Matrix::drawHistogram(const float * histogram, int x, int y, int value, int nbins, int psize)
{
  int binwidth = nbins > (psize>>1) ? 1 : 2;
  if(histogram == NULL)
    return;
  for (int n = 0; n < nbins; ++n)
  {
    int binheight = std::min(psize, std::max(0, (int)round(histogram[n] * psize)));
    for(int ix = 0; ix < binwidth; ++ix)
      for(int iy = 0; iy < binheight; ++iy)
        (*this)(x+ix+n*binwidth, y+psize-1-iy) = value;
  }
}

void Matrix::drawNumber(int x, int y, int n, int value)
{
  bool chars[10][4*7] = {
    {0,1,1,0, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 0,1,1,0}, //0
    {0,1,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0, 0,1,1,1}, //1
    {0,1,1,0, 1,0,0,1, 0,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0, 1,1,1,1}, //2
    {0,1,1,0, 1,0,0,1, 0,0,0,1, 0,1,1,0, 0,0,0,1, 1,0,0,1, 1,1,1,0}, //3
    {0,0,1,0, 0,1,1,0, 1,0,1,0, 1,0,1,0, 1,1,1,1, 0,0,1,0, 0,0,1,0}, //4
    {1,1,1,1, 1,0,0,0, 1,0,0,0, 1,1,1,0, 0,0,0,1, 0,0,0,1, 1,1,1,0}, //5
    {0,1,1,1, 1,1,0,0, 1,0,0,0, 1,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,0}, //6
    {1,1,1,1, 0,0,0,1, 0,0,0,1, 0,0,1,0, 0,0,1,0, 0,1,0,0, 0,1,0,0}, //7
    {0,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,0}, //8
    {0,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,1, 0,0,0,1, 0,0,1,1, 1,1,1,0}}; //9
  int a = abs(n), tx = -4;
  do {
    // draw digits from right to left
    int c = a%10;
    for (int i = 0; i < 4*7; i++)
      if (chars[c][i])
      {
        int tmpx = x + tx + (i%4), tmpy = y + (i>>2);
        if (tmpx >= 0 && tmpy >= 0 && tmpx < ivWidth && tmpy < ivHeight)
          (*this)(tmpx, tmpy) = value;   
      }
    tx -= 5;
    a = a/10;
  } while(a > 0);
  if (n < 0)
    // draw a minus sign
    for (int i = 1; i < 4; i++){
      int tmpx = x + tx + i, tmpy = y + 3;
      if (tmpx >= 0 && tmpy >= 0 && tmpx < ivWidth && tmpy < ivHeight)
        (*this)(tmpx, tmpy) = value;   
    }
}

Matrix& Matrix::operator=(const Matrix& copyFrom)
{
  if (this != &copyFrom) {
    if (ivData != 0) delete[] ivData;
    ivWidth = copyFrom.ivWidth;
    ivHeight = copyFrom.ivHeight;
    if (copyFrom.ivData == 0) ivData = 0;
    else {
      int wholeSize = ivWidth*ivHeight;
      ivData = new float[wholeSize];
      memcpy(ivData, copyFrom.ivData, wholeSize * sizeof(float));
      //for (register int i = 0; i < wholeSize; i++)
      //  ivData[i] = copyFrom.ivData[i];
    }
  }
  return *this;
}

Matrix& Matrix::operator+=(const float value)
{
  int wholeSize = ivWidth*ivHeight;
  for (int i = 0; i < wholeSize; i++)
    ivData[i] += value;
  return *this;
}

Matrix& Matrix::operator*=(const float value)
{
  int wholeSize = ivWidth*ivHeight;
  for (int i = 0; i < wholeSize; i++)
    ivData[i] *= value;
  return *this;
}

float Matrix::avg() const
{
  float aAvg = 0;
  int aSize = ivWidth*ivHeight;
  for (int i = 0; i < aSize; i++)
    aAvg += ivData[i];
  return aAvg/aSize;
}

float Matrix::norm2() const
{
  double sqSum = 0;
  int aSize = ivWidth*ivHeight;
  for (int i = 0; i < aSize; i++)
    sqSum += ivData[i]*ivData[i];
  return sqSum;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
  if (m1.xSize() != m2.ySize()){
    std::cerr << "cannot multiply incompatible matrices!" << std::endl;
    return Matrix();
  }
    
  Matrix result(m2.xSize(),m1.ySize(),0);
  for (int y = 0; y < result.ySize(); y++)
    for (int x = 0; x < result.xSize(); x++)
      for (int i = 0; i < m1.xSize(); i++)
        result(x,y) += m1(i,y)*m2(x,i);
  return result;
}

void writePPM(const char* filename, const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix)
{
  FILE* outimage = fopen(filename, "wb");
  int width = rMatrix.xSize(), height = rMatrix.ySize();
  fprintf(outimage, "P6 \n");
  fprintf(outimage, "%d %d \n255\n", width, height);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
      unsigned char tmp = (unsigned char)rMatrix(x,y);
      fwrite (&tmp, sizeof(unsigned char), 1, outimage);
      tmp = (unsigned char)gMatrix(x,y);
      fwrite (&tmp, sizeof(unsigned char), 1, outimage);
      tmp = (unsigned char)bMatrix(x,y);
      fwrite (&tmp, sizeof(unsigned char), 1, outimage);
    }
  fclose(outimage);
}

/* ----------------------------------------------------------
 *           Stuff for (fast) Summed AreaTables             *
 * ---------------------------------------------------------*/
