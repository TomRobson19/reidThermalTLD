/* Copyright (C) 2012 Christian Lutz, Thorsten Engesser
 * 
 * This file is part of motld
 * 
 * Some parts of this implementation are based
 * on materials to a lecture by Thomas Brox
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#ifdef GNU_COMPILER
  #include <strstream>
#else
  #include <sstream>
#endif

#ifndef PI
#define PI 3.1415926536
#endif

#ifndef round
#define round(x) floor(x + 0.5)
#endif

#define MAXF(a,b,c,d) MAX(MAX(a,b),MAX(c,d))
#define MINF(a,b,c,d) MIN(MIN(a,b),MIN(c,d))
#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)
#endif

/// datastructure linking objects to their (possible) location
struct ObjectBox
{
  /// x-component of top left coordinate
  float x;
  /// y-component of top left coordinate
  float y;
  /// width of the image section
  float width;
  /// height of the image section
  float height;
  /// identifies object, which is represented by ObjectBox
  int objectId;
};

/// datastructure for images (greyscale or single color)
class Matrix {
public:
  /// Default constructor
  inline Matrix();
  /// Constructor
  inline Matrix(const int width, const int height);
  /// Copy constructor
  Matrix(const Matrix& copyFrom);
  /// Constructor with implicit filling
  Matrix(const int width, const int height, const float value);
  /// Destructor
  virtual ~Matrix();

  /// fills the matrix from a char-array (size has to be already set)
  void copyFromCharArray(unsigned char * source);
  /// fills the matrix from a float-array (for a given size)
  void copyFromFloatArray(float * source, int srcwidth, int width, int height);
  /// fills the matrix from a sub-part of a float-array
  void copyFromFloatArray(const float * const source, int srcwidth, int srcheight, int x, int y, int width, int height);
  /// Creates a grayscale matrix out of r, g, and b matrices
  void fromRGB(const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix);
  /// Creates a grayscale matrix out of an array [r0, r1, ..., g0, g1, ..., b0, b1, ...]
  void fromRGB(unsigned char * source);
  /// Computes derivative in x direction (result will be in result)
  void derivativeX(Matrix& result) const;
  /// Computes derivative in y direction (result will be in result)
  void derivativeY(Matrix& result) const;
  /// Applies 3x3 Scharr filter in x direction (result will be in result)
  void scharrDerivativeX(Matrix& result) const;
  /// Applies 3x3 Scharr filter in y direction (result will be in result)
  void scharrDerivativeY(Matrix& result) const;
  /// Applies 3x3 Sobel filter in x direction (result will be in result)
  void sobelDerivativeX(Matrix& result) const;
  /// Applies 3x3 Sobel filter in y direction (result will be in result)
  void sobelDerivativeY(Matrix& result) const;
  /// Applies a Gaussian filter
  void gaussianSmooth(const float sigma, const int filterSize = 0);
  /// Saves the matrix as a picture in pgm-Format
  void writeToPGM(const char *filename) const;
  /// Returns a patch around the central point using bilinear interpolation
  Matrix getRectSubPix(float centerx, float centery, int width, int height) const;
 
  /// Changes the size of the matrix, data will be lost
  void setSize(int width, int height);
  /// Downsamples image to half of its size (result will be in result)
  void halfSizeImage(Matrix& result) const;
  /// Downsamples the matrix
  void downsample(int newWidth, int newHeight);
  /// Downsamples the matrix using bilinear interpolation
  void downsampleBilinear(int newWidth, int newHeight);  
  /// Upsamples the matrix
  void upsample(int newWidth, int newHeight);
  /// Upsamples the matrix using bilinear interpolation
  void upsampleBilinear(int newWidth, int newHeight);
  /// Scales the matrix (includes upsampling and downsampling)
  void rescale(int newWidth, int newHeight);
  
  /// Fills the matrix with the value value (see also operator =)
  void fill(const float value);
  /// Copies a rectangular part from the matrix into result, the size of result will be adjusted
  void cut(Matrix& result,const int x1, const int y1, const int x2, const int y2);
  /// Clips values that exceed the given range
  void clip(float aMin, float aMax);
  /// Inverts a 3x3 matrix
  void inv3();

  // Some drawing utilities
  /// Draws a line into the image
  void drawLine(int x1, int y1, int x2, int y2, float value = 255);
  /// Draws a Cross
  void drawCross(int x, int y, int value = 255, int crossSize = 1);
  /// Draws an ObjectBox into the image
  void drawBox(ObjectBox b, int value = 255);
  /// Draws a dashed ObjectBox into the image
  void drawDashedBox(ObjectBox b, int value = 255, int dashLength = 3, bool dotted = false);
  /// Draws a NN-Patch at position (x,y)
  void drawPatch(const Matrix & b, int x, int y, float avg = 0);
  /// Draws a histogram at position (x,y)
  void drawHistogram(const float * histogram, int x, int y, int value = 255, int nbins = 7, int psize = 15);
  /// Prints a number at position (x,y)
  void drawNumber(int x, int y, int n, int value = 255);
    
  /// Gives full access to matrix values
  inline float& operator()(const int ax, const int ay) const;
  /// Fills the matrix with the value value (equivalent to fill())
  inline Matrix& operator=(const float value);
  /// Copies the matrix copyFrom to this matrix (size of matrix might change)
  Matrix& operator=(const Matrix& copyFrom);
  /// Adds a constant to the matrix
  Matrix& operator+=(const float value);
  /// Multiplication with a scalar
  Matrix& operator*=(const float value);

  /// Returns the average value
  float avg() const;
  /// Returns the squared norm (i.e. sum of squared values)
  float norm2() const;
  /// Returns the width of the matrix
  inline int xSize() const;
  /// Returns the height of the matrix
  inline int ySize() const;
  /// Returns the size (width*height) of the matrix
  inline int size() const;
  /// Gives access to the internal data representation
  inline float* data() const;
  
  /// Performs an affine warping of an image section
  Matrix affineWarp(const Matrix & t, const ObjectBox & b, const bool & preservear) const;
  /// Creates a warp matrix for scaling / roatating
  static Matrix createWarpMatrix(const float& angle, const float& scale);
  /// Creates an Integral Image
  float* createSummedAreaTable() const;
  /// Creates an Integral Image and an Integral Image of squared values
  float** createSummedAreaTable2() const;
  
protected:
  int ivWidth, ivHeight;
  float *ivData;
};

/// Matrix product
Matrix operator*(const Matrix& m1, const Matrix& m2);
/// Provides basic output functionality (only appropriate for small matrices)
std::ostream& operator<<(std::ostream& aStream, const Matrix& aMatrix);

/// Outputs an RGB image in PPM format
void writePPM(const char* filename, const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix);

inline Matrix::Matrix()
{
  ivData = NULL; 
  ivWidth = ivHeight = 0;
}

inline Matrix::Matrix(const int width, const int height)
  : ivWidth(width), ivHeight(height)
{
  ivData = new float[width*height];
}

inline float& Matrix::operator()(const int ax, const int ay) const
{
  #ifdef _DEBUG
    if (ax >= ivWidth || ay >= ivHeight || ax < 0 || ay < 0){
      std::cerr << "Exception EMatrixRangeOverflow: x = " << ax << ", y = " << ay << std::endl;
      return 0;
    }
  #endif
  return ivData[ivWidth*ay+ax];
}

inline Matrix& Matrix::operator=(const float value)
{
  fill(value);
  return *this;
}

inline int Matrix::xSize() const {
  return ivWidth;
}

inline int Matrix::ySize() const {
  return ivHeight;
}

inline int Matrix::size() const {
  return ivWidth*ivHeight;
}

inline float* Matrix::data() const {
  return ivData;
}


inline float* Matrix::createSummedAreaTable() const
{  
  int width = ivWidth + 1;
  int height = ivHeight + 1;

  float* sat = new float[width*height];

  for (int x = 0; x < width; ++x)
    sat[x] = 0;

  int n = 0;
  for (int y = 1; y < height; ++y)
  {
    int yoffset = y * width;
    sat[yoffset] = 0;
    for (int x = 1; x < width; ++x, ++n)
    {
      int offset = yoffset + x;
      sat[offset] = ivData[n] + sat[offset-1] + sat[offset-width] - sat[offset-width-1];
    }
  }

  return sat;
}

inline float** Matrix::createSummedAreaTable2() const
{  
  int width = ivWidth + 1;
  int height = ivHeight + 1;

  float* sat = new float[width*height];
  float* sat2 = new float[width*height];

  for (int x = 0; x < width; ++x)
  sat[x] = sat2[x] = 0;

  int n = 0;
  for (int y = 1; y < height; ++y)
  {
    int yoffset = y * width;
    sat[yoffset] = sat2[yoffset] = 0;
    for (int x = 1; x < width; ++x, ++n)
    {
      int offset = yoffset + x;
      sat[offset] = ivData[n] + sat[offset-1] + sat[offset-width] - sat[offset-width-1];
      sat2[offset] = ivData[n]*ivData[n] + sat2[offset-1] + sat2[offset-width] - sat2[offset-width-1];
    }
  }

  float** result = new float*[2];
  result[0] = sat;
  result[1] = sat2;
  return result;
}

inline double summedTableArea(float* sat, int width, int x1, int y1, int x2, int y2)
{
  ++width; ++x2; ++y2;
  return sat[y2*width+x2] - sat[y1*width+x2] - sat[y2*width+x1] + sat[y1*width+x1];
}

inline double summedTableArea(const float * const sat, int * indices)
{
  return sat[indices[0]] - sat[indices[1]] - sat[indices[2]] + sat[indices[3]];
}

inline int* getSATIndices(int width, int x1, int y1, int x2, int y2)
{
  ++width; ++x2; ++y2;
  int* result = new int[4];
  result[0] = y2*width+x2;
  result[1] = y1*width+x2;
  result[2] = y2*width+x1;
  result[3] = y1*width+x1;
  return result;
}

inline void getSATIndices(int * array, int width, int x1, int y1, int x2, int y2)
{
  ++width; ++x2; ++y2;
  array[0] = y2*width+x2;
  array[1] = y1*width+x2;
  array[2] = y2*width+x1;
  array[3] = y1*width+x1;
}

inline int* getSATIndices(int width, int boxw, int boxh)
{
  return getSATIndices(width,0,0,boxw-1,boxh-1);
}

/* ----------------------------------------------------------
 *                Stuff for affine warping                  *
 * ---------------------------------------------------------*/

inline Matrix Matrix::affineWarp(const Matrix& t, const ObjectBox& b, const bool& preservear) const
{
  float widthHalf = b.width / 2;
  float heightHalf = b.height / 2;
 
  // object space transformation
  Matrix ost(3,3);
  ost.ivData[0] = 1; ost.ivData[1] = 0; ost.ivData[2] = b.x + widthHalf - 0.5;
  ost.ivData[3] = 0; ost.ivData[4] = 1; ost.ivData[5] = b.y + heightHalf - 0.5;
  ost.ivData[6] = 0; ost.ivData[7] = 0; ost.ivData[8] = 1;
  
  Matrix ostinv = ost;
  ostinv.ivData[2] = - ostinv.ivData[2];
  ostinv.ivData[5] = - ostinv.ivData[5];
  Matrix tinv = t; tinv.inv3();
  
  Matrix trans = ost * tinv * ostinv;
  
  Matrix result(b.width, b.height);
  for (int dx = 0; dx <= b.width-1; ++dx)
  {
    for (int dy = 0; dy <= b.height-1; ++dy)
    { 
      Matrix v(1,3);
      float x = b.x + dx;
      float y = b.y + dy;
      v.ivData[0] = x; v.ivData[1] = y; v.ivData[2] = 1;
      v = trans * v;
      
      int x1 = MAX(0,MIN(ivWidth-1,floor(v.ivData[0]))); int x2 = MAX(0,MIN(ivWidth-1,ceil(v.ivData[0])));
      int y1 = MAX(0,MIN(ivHeight-1,floor(v.ivData[1]))); int y2 = MAX(0,MIN(ivHeight-1,ceil(v.ivData[1])));
      double dx1 = v.ivData[0] - x1; double dy1 = v.ivData[1] - y1;
      
      result(dx, dy) =
               (1-dx1) * ((1-dy1) * (*this)(x1, y1) + dy1 * (*this)(x1,y2))
                 + dx1 * ((1-dy1) * (*this)(x2, y1) + dy1 * (*this)(x2,y2));
    }
  }  
  return result; 
}

inline Matrix Matrix::createWarpMatrix(const float& angle, const float& scale)
{
  Matrix scm(3,3); 
  scm(0, 0) = scale; scm(1, 0) =     0; scm(2, 0) = 0;
  scm(0, 1) =     0; scm(1, 1) = scale; scm(2, 1) = 0;
  scm(0, 2) =     0; scm(1, 2) =     0; scm(2, 2) = 1;
  Matrix anm(3,3);
  float ca = cos(angle); float sa = sin(angle);
  anm(0, 0) =  ca; anm(1, 0) =  sa; anm(2, 0) = 0;
  anm(0, 1) = -sa; anm(1, 1) =  ca; anm(2, 1) = 0;
  anm(0, 2) =   0; anm(1, 2) =   0; anm(2, 2) = 1;
  Matrix wm = anm * scm;
  return wm;
}

/* ----------------------------------------------------------
 *                  box overlap checking                    *
 * ---------------------------------------------------------*/

inline float rectangleOverlap( float minx1, float miny1,
    float maxx1, float maxy1, float minx2, float miny2,
    float maxx2, float maxy2 )
{
  if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2)
  {
    return 0.0f;
  }
  else
  {
    float dx = MIN(maxx2, maxx1)-MAX(minx2, minx1);
    float dy = MIN(maxy2, maxy1)-MAX(miny2, miny1);
    float area1 = (maxx1-minx1)*(maxy1-miny1);
    float area2 = (maxx2-minx2)*(maxy2-miny2);
    float avgarea = 0.5 * (area1+area2);
    float overlaparea = dx*dy;
    return overlaparea/avgarea;
  }
}

inline float rectangleOverlap(const ObjectBox& a, const ObjectBox& b)
{
  return rectangleOverlap(a.x, a.y, a.x+a.width, a.y+a.height, 
                          b.x, b.y, b.x+b.width, b.y+b.height );
}

#endif
